#!/usr/bin/env python3
import argparse
import base64, hashlib
import os,sys,pdb
import shutil, subprocess
import urllib.request
import zipfile




DOCS = """
Key to understanding the mess below is that a Python wheel (.whl) is a zip
archive with a particular set of files inside, with particular names and
particular contents.

There's a main folder jaxlib/, plus jaxlib-version+buildtag.dist-info/ with
metadata about the package. To prevent corruption^H^H^H^H^H^H easy
manipulation, this metadata includes a file called RECORD with the hashes and
lengths of all the contents (excluding itself, since no file can contain its
own hash).

So, to perform patching of any file, one must adjust the corresponding RECORD
entry as well.

Some platforms, such as Compute Canada clusters, are configured to reject
wheels tagged as "manylinux1" or "manylinux2010". There is a good reason why
this is so: Compute Canada clusters are not, in fact, manylinux compliant.
Vulgar externally-sourced Python wheels of any kind require patching to work
on those clusters.

To work around that limitation, another file in .dist-info/ called WHEEL
must be edited to rewrite its self-declared Tag: entry, and then RECORD
must be updated as well.

This script, therefore, offers the ability to:

 - Download
 - Binary-patch
 - Retag

these jaxlib wheels.
"""



def vercmp(a, b):
    a = tuple(map(int, a.split(".")))
    b = tuple(map(int, b.split(".")))
    if   a<b:
        return -1
    elif a>b:
        return +1
    return 0

def getrecordline(arcname, arcdata):
    m = hashlib.sha256()
    m.update(arcdata)
    d = base64.urlsafe_b64encode(m.digest()).decode().strip("=")
    return f"{arcname},sha256={d},{len(arcdata)}"

def arebinarypatching(args):
    return bool(args.add_origin or args.set_runpath is not None)

def getfilename(args, output=False):
    jx_tag = args.jaxlib_version
    cu_tag = args.cuda_version
    py_tag = args.python_version
    if vercmp(jx_tag, "0.1.49") <= 0:
        lx_tag = "linux"
    else:
        lx_tag = "manylinux2010"
    lx_tag = (output and args.tag) or lx_tag
    return f"jaxlib-{jx_tag}+{cu_tag}-{py_tag}-none-{lx_tag}_x86_64.whl"

def getfileurl(args):
    return f"https://storage.googleapis.com/jax-releases/{args.cuda_version}/{getfilename(args)}"

def download(args):
    dirpath  = os.path.join(args.work, "original")
    os.makedirs(dirpath, mode=0o755, exist_ok=True)
    filename = getfilename(args)
    fileurl  = getfileurl(args)
    filepath = os.path.join(dirpath, filename)
    if os.path.isfile(filepath):
        if not args.force:
            return
        os.remove(filepath)
    urllib.request.urlretrieve(fileurl, filepath)

def patch(args):
    dirpath = os.path.join(args.work, "unpacked")
    os.makedirs(os.path.join(args.work, "unpacked"), mode=0o755, exist_ok=True)
    os.makedirs(os.path.join(args.work, "repacked"), mode=0o755, exist_ok=True)
    
    ifilename = getfilename(args)
    ifilepath = os.path.join(args.work, "original", ifilename)
    ifilemode = "r"
    ofilename = getfilename(args, output=True)
    ofilepath = os.path.join(args.work, "repacked", ofilename)
    ofilemode = "w" if args.force else "x"
    
    retagging = (ifilename != ofilename)
    binarypatching = arebinarypatching(args)
    
    if binarypatching:
        runpath = []
        if args.add_origin:
            runpath += ["$ORIGIN"]
        if args.set_runpath:
            runpath += [args.set_runpath]
        runpath = ":".join(runpath)
        cmd  = ["patchelf", "--set-rpath", runpath]
    
    with zipfile.ZipFile(ifilepath, ifilemode) as zi, \
         zipfile.ZipFile(ofilepath, ofilemode, zipfile.ZIP_DEFLATED) as zo:
        ZRECORDS = {}
        ZRECORDINFO = None
        for ze in zi.infolist():
            if ze.filename.endswith(".dist-info/RECORD"):
                #
                # RECORD is the file containing all the checksums. We can't write it until we
                # have done all the patching work. Hold it back.
                #
                # Self-evidently, RECORD cannot contain its own checksum, so it has a special
                # unprotected record of itself.
                #
                ZRECORDINFO = ze
                continue
            
            if ze.filename.endswith(".dist-info/WHEEL") and retagging:
                #
                # WHEEL is a file containing a tag that needs to be rewritten when retagging.
                # It is protected by a checksum in RECORD. Rewrite this file in memory without
                # extracting to a filesystem.
                #
                WHEEL = ""
                for l in zi.read(ze).decode().split("\n"):
                    if l.startswith("Tag: ") and args.python_version in l:
                        WHEEL += f"Tag: {args.python_version}-none-{args.tag}_x86_64\n"
                    else:
                        WHEEL += l+"\n"
                WHEEL = WHEEL.encode()
                
                del ze.file_size
                del ze.compress_size
                zo.writestr(ze, WHEEL)
                ZRECORDS[ze.filename] = getrecordline(ze.filename, WHEEL)
                continue
            
            if ze.filename.endswith(".so") and binarypatching:
                #
                # Okay, assume this is a shared library.
                #   1. Extract file to filesystem
                #   2. Chmod to 755, because by default they are packaged 555 (strange, yes)
                #   3. patchelf, which modifies the binary in-place
                #   4. Insert modified data into zip-file
                #   5. Compute RECORD line.
                #   6. Delete file from filesystem.
                #
                ufilepath = zi.extract(ze, dirpath)
                os.chmod(ufilepath, 0o755)
                subprocess.check_call(cmd + [ufilepath])
                filedata = open(ufilepath, "rb").read()
                del ze.file_size
                del ze.compress_size
                zo.writestr(ze, filedata)
                ZRECORDS[ze.filename] = getrecordline(ze.filename, filedata)
                os.remove(ufilepath)
                continue
            
            #
            # Otherwise this was not a file that needed patching. Simply copy it over.
            #
            zo.writestr(ze, zi.read(ze))
        
        
        # Handle RECORD file.
        if ZRECORDINFO is None:
            print("WARN: Very strange wheel, it has no RECORD... Unlikely to work...")
        else:
            RECORD = ""
            for l in zi.read(ZRECORDINFO).decode().split("\n"):
                if ",sha256=" not in l:
                    RECORD += l + "\n"
                    continue
                
                arcfilename = l.split(",sha256=")[0]
                if arcfilename and arcfilename in ZRECORDS:
                    RECORD += ZRECORDS[arcfilename] + "\n"
                    continue
                
                RECORD += l + "\n"
            
            RECORD = RECORD[:-1].encode()
            del ZRECORDINFO.file_size
            del ZRECORDINFO.compress_size
            zo.writestr(ZRECORDINFO, RECORD)




if __name__ == "__main__":
    # Create parser
    argp = argparse.ArgumentParser(epilog=DOCS, formatter_class=argparse.RawDescriptionHelpFormatter)
    argp.add_argument("--jaxlib-version", "-V", default="0.1.57")
    argp.add_argument("--cuda-version",   "-C", default="cuda101")
    argp.add_argument("--python-version", "-P", default="cp37")
    argp.add_argument("--download",       "-d", action="store_true",  dest="download", default=True)
    argp.add_argument("--no-download",          action="store_false", dest="download", default=True)
    argp.add_argument("--patch",          "-p", action="store_true",  dest="patch",    default=True)
    argp.add_argument("--no-patch",             action="store_false", dest="patch",    default=True)
    argp.add_argument("--retag",          "-t", type=str, default=None, dest="tag")
    argp.add_argument("--add-origin",           action="store_true")
    argp.add_argument("--set-runpath",          default=None)
    argp.add_argument("--work",           "-w", default="./tmp")
    argp.add_argument("--force",          "-f", action="store_true")
    
    
    
    # Parse arguments
    args = argp.parse_args(sys.argv[1:])
    if "." in args.cuda_version:
        # Fixup if CUDA    version is specified as "10.0.130" or somesuch
        args.cuda_version   = "cuda"+"".join(args.cuda_version  .split(".")[:2])
    if "." in args.python_version:
        # Fixup if CPython version is specified as "3.6.12"   or somesuch
        args.python_version = "cp"  +"".join(args.python_version.split(".")[:2])
    
    
    # Fail-fast, early check
    if args.patch and arebinarypatching(args):
        if not shutil.which("patchelf"):
            print("ERR: Need patchelf in the PATH for binary-patching work!")
            sys.exit(1)
    
    
    
    # Act upon them
    if args.download:
        download(args)
    if args.patch:
        patch(args)
