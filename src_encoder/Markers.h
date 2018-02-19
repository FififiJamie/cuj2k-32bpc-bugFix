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

#ifndef MARKERS_H
#define MARKERS_H

/*Delimiting markers and marker segments */
/* Start of codestream (SOC): 0xFF4F */
#define SOC 0xff4f
/* Start of tile-part (SOT): 0xFF90 */
#define SOT 0xff90

    /* Start of data (SOD): 0xFF93 */
#define SOD 0xff93

    /* End of codestream (EOC): 0xFFD9 */
#define EOC 0xffd9
/*Fixed information marker segments */
/* SIZ marker (Image and tile size): 0xFF51 */
#define SIZ 0xff51
/* No special capabilities (baseline) in codestream, in Rsiz field of SIZ
 * marker: 0x00. All flag bits are turned off */
#define RSIZ_BASELINE 0x00
/* Error resilience marker flag bit in Rsiz field in SIZ marker: 0x01 */
#define RSIZ_ER_FLAG 0x01
/* ROI present marker flag bit in Rsiz field in SIZ marker: 0x02 */
#define RSIZ_ROI 0x02
/* Component bitdepth bits in Ssiz field in SIZ marker: 7 */
#define SSIZ_DEPTH_BITS 7
/* The maximum number of component bitdepth */
#define MAX_COMP_BITDEPTH 38

/*Functional marker segments*/

/* COD/COC marker */

/* Coding style default (COD): 0xFF52 */
#define COD 0xff52

/* Coding style component (COC): 0xFF53 */
#define COC 0xff53

/* Precinct used flag */
#define SCOX_PRECINCT_PARTITION 1
/* Use start of packet marker */
#define SCOX_USE_SOP 2
/* Use end of packet header marker */
#define SCOX_USE_EPH 4
/* The default size exponent of the precincts */
#define PRECINCT_PARTITION_DEF_SIZE = 0xffff

/* RGN marker segment */
/*Region-of-interest (RGN): 0xFF5E */
#define RGN 0xff5e

/* Implicit (i.e. max-shift) ROI flag for Srgn field in RGN marker
 * segment: 0x00 */
#define SRGN_IMPLICIT = 0x00;

/* QCD/QCC markers */
/* Quantization default (QCD): 0xFF5C */
#define QCD 0xff5c

/* Quantization component (QCC): 0xFF5D */
#define QCC 0xff5d
/* Guard bits shift in SQCX field: 5 */
#define SQCX_GB_SHIFT 5
/* Guard bits mask in SQCX field: 7 */
#define SQCX_GB_MSK 7
/* No quantization (i.e. embedded reversible) flag for Sqcd or Sqcc
 * (Sqcx) fields: 0x00. */
#define SQCX_NO_QUANTIZATION 0x00
/* Scalar derived (i.e. LL values only) quantization flag for Sqcd or
 * Sqcc (Sqcx) fields: 0x01. */
#define SQCX_SCALAR_DERIVED 0x01
/* Scalar expounded (i.e. all values) quantization flag for Sqcd or Sqcc
 * (Sqcx) fields: 0x02. */
#define SQCX_SCALAR_EXPOUNDED 0x02
/* Exponent shift in SPQCX when no quantization: 3 */
#define SQCX_EXP_SHIFT 3
/* Exponent bitmask in SPQCX when no quantization: 3 */
#define SQCX_EXP_MASK = (1<<5)-1
/* The "SOP marker segments used" flag within Sers: 1 */
#define ERS_SOP 1
/* The "segmentation symbols used" flag within Sers: 2 */
#define ERS_SEG_MARKERS 2

    /* Progression order change */
#define POC 0xff5f

/* Pointer marker segments */
/* Tile-part lengths (TLM): 0xFF55 */
#define TLM 0xff55
/* Packet length, main header (PLM): 0xFF57 */
#define PLM 0xff57
/* Packet length, tile-part header (PLT): 0xFF58 */
#define PLT 0xff58
/* Packed packet headers, main header (PPM): 0xFF60 */
#define PPM 0xff60
/* Packed packet headers, tile-part header (PPT): 0xFF61 */
#define PPT 0xff61
/* Maximum length of PPT marker segment */
#define MAX_LPPT 65535
/* Maximum length of PPM marker segment */
#define MAX_LPPM 65535
/* In bit stream markers and marker segments */
/* Start pf packet (SOP): 0xFF91 */
#define SOP 0xff91
 /* Length of SOP marker (in bytes) */
#define SOP_LENGTH 6
/* End of packet header (EPH): 0xFF92 */
#define EPH 0xff92
/* Length of EPH marker (in bytes) */
#define EPH_LENGTH 2
/* Informational marker segments */
/* Component registration (CRG): 0xFF63 */
#define CRG 0xff63
/* Comment (COM): 0xFF64 */
#define COM 0xff64
/* General use registration value (COM): 0x0001 */
#define RCOM_GEN_USE 0x0001


#endif
