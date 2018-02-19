/* CUJ2K - JPEG2000 Encoder on CUDA
http://cuj2k.sourceforge.net/

Copyright (c) 2009 Norbert Fuerst, Martin Heide, Armin Weiss, Simon Papandreou, Ana Balevic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE. */

#ifndef MISC_H
#define MISC_H

/*#ifndef min
#define min(a,b) (((a)<(b)) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) (((a)>(b)) ? (a) : (b))
#endif*/

#define square(x) ((x)*(x))


#if defined(_DEVICEEMU) && (!defined(NDEBUG))
#define my_dev_assert(cond) ((!!(cond)) || printf("Assert '%s' failed; file %s line %d\n", (#cond), __FILE__, __LINE__))
#else
#define my_dev_assert(cond) ((void)0)
#endif

#endif
