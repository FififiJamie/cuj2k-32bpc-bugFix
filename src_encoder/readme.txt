CUJ2K 1.1 - JPEG2000 Encoder on CUDA
http://cuj2k.sourceforge.net/
Copyright (c) 2009 Norbert Fuerst, Martin Heide, Armin Weiss, 
Simon Papandreou, Ana Balevic



Compilation
===========

Windows: ------------------------------------------------------------
- use src_encoder\gpu_vc2005.sln for Visual Studio 2005
- use src_encoder\gpu_vc08.sln for Visual Studio 2008

Remember to choose the correct configuration (Release for GPU code,
EmuRelease for CPU emulation code)!

- use src_gui\dotnet_gui.sln to compile the .Net GUI
  (currently only for Visual Studio 2008)


Linux: --------------------------------------------------------------

$ cd src_encoder
$ make
optional: (execute as root)
$ make install   (copies the executable to /usr/local/bin/)

For cleaning up:
$ make clean

You may use the options emu=1 and dbg=1 to enable debug 
and/or emulation; e.g.:

$ make emu=1 dbg=1
$ make install emu=1 dbg=1
(you must use the same options in both commands, also for make clean)

The executable and temporary files are stored in subdirectories depending
on the supplied options.


**** **** ***** **** **** **** ***** **** **** **** ***** ****
Linux troubleshooting:

1 ) When we tested it, compilation only worked with gcc 4.3 and not
with gcc 4.4; but this might be fixed by NVIDIA in the meantime, so
check for the latest Toolkit and SDK versions.
If you can't compile, install gcc and g++ 4.3. Two options:
a) uninstall gcc and g++ 4.4 before installing version 4.3
b) after installing version 4.3, ensure that the commands gcc and
   g++ correspond to version 4.3. Should work about like this
   (might be different on your system, no guarantee!)

     (become root)
   $ cd /usr/bin
   $ rm gcc g++
   $ ln -s gcc-4.3 gcc
   $ ln -s g++-4.3 g++
   $ chmod a+x gcc g++



2 ) If you get runtime errors like "... cudaGetDeviceProperties not implemented",
you probably have an older CUDA version. You need to recompile cuj2k:
Uncomment the line "COMMONFLAGS += -DNO_DEVICE_PROP" in the makefile, then run
$ make clean
$ make


3 ) If the executable from the binary package does not work, and you get
other error messages not mentioned here, just try compiling it by yourself, 
because then probably some library versions do not match.

**** **** ***** **** **** **** ***** **** **** **** ***** ****
