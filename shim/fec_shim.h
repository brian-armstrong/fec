/*
 * fec_shim.h - C declarations for the fec-shim library.
 *
 * This header declares the subset of Phil Karn's libfec C ABI that fec-shim
 * implements. Link against the shim's libfec.so or libfec.a and include this
 * header (or libfec's own fec.h) to call fec's Rust codecs from C.
 *
 * The signatures match libfec so that code written against libfec can use this
 * library unchanged, as long as it only calls the functions declared here.
 */

#ifndef FEC_SHIM_H
#define FEC_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Convolutional generator polynomials, matching libfec. Each decoder below is
 * built with these, so they are provided for code that references them.
 */
#define V27POLYA 0x6d
#define V27POLYB 0x4f

#define V29POLYA 0x1af
#define V29POLYB 0x11d

#define V39POLYA 0x1ed
#define V39POLYB 0x19b
#define V39POLYC 0x127

#define V615POLYA 042631
#define V615POLYB 047245
#define V615POLYC 056507
#define V615POLYD 073363
#define V615POLYE 077267
#define V615POLYF 064537

/*
 * Convolutional (Viterbi) decoders. Each code has the libfec lifecycle:
 * create, init, update in blocks, chainback, delete. create returns an opaque
 * decoder handle. update_..._blk feeds soft symbols. chainback writes the
 * decoded bits. init and update return an int for libfec compatibility.
 */

/* rate 1/2, k=7 */
void *create_viterbi27(int num_decoded_bits);
void delete_viterbi27(void *vit);
int init_viterbi27(void *vit, int starting_state);
int update_viterbi27_blk(void *vit, const unsigned char *encoded, int num_groups);
int chainback_viterbi27(void *vit, unsigned char *decoded, unsigned int num_bits, int end_state);

/* rate 1/2, k=9 */
void *create_viterbi29(int num_decoded_bits);
void delete_viterbi29(void *vit);
int init_viterbi29(void *vit, int starting_state);
int update_viterbi29_blk(void *vit, const unsigned char *encoded, int num_groups);
int chainback_viterbi29(void *vit, unsigned char *decoded, unsigned int num_bits, int end_state);

/* rate 1/3, k=9 */
void *create_viterbi39(int num_decoded_bits);
void delete_viterbi39(void *vit);
int init_viterbi39(void *vit, int starting_state);
int update_viterbi39_blk(void *vit, const unsigned char *encoded, int num_groups);
int chainback_viterbi39(void *vit, unsigned char *decoded, unsigned int num_bits, int end_state);

/* rate 1/6, k=15 */
void *create_viterbi615(int num_decoded_bits);
void delete_viterbi615(void *vit);
int init_viterbi615(void *vit, int starting_state);
int update_viterbi615_blk(void *vit, const unsigned char *encoded, int num_groups);
int chainback_viterbi615(void *vit, unsigned char *decoded, unsigned int num_bits, int end_state);

/*
 * Reed-Solomon over GF(2^8). init_rs_char builds a decoder for arbitrary
 * parameters. The _8 and _ccsds calls are the fixed CCSDS (255,223) code.
 * decode calls return the number of symbols corrected, or -1 if the block
 * could not be corrected.
 */

/* general Reed-Solomon */
void *init_rs_char(int symbol_size, int primitive_polynomial, int first_consecutive_root,
                   int root_gap, int number_roots, unsigned int pad);
void free_rs_char(void *rs);
void encode_rs_char(void *rs, const unsigned char *msg, unsigned char *parity);
int decode_rs_char(void *rs, unsigned char *block, const int *erasure_locations, int num_erasures);

/* CCSDS (255,223), conventional basis */
void encode_rs_8(const unsigned char *data, unsigned char *parity, int pad);
int decode_rs_8(unsigned char *data, const int *erasure_locations, int num_erasures, int pad);

/* CCSDS (255,223), dual (Berlekamp) basis */
void encode_rs_ccsds(const unsigned char *data, unsigned char *parity, int pad);
int decode_rs_ccsds(unsigned char *data, const int *erasure_locations, int num_erasures, int pad);

#ifdef __cplusplus
}
#endif

#endif /* FEC_SHIM_H */
