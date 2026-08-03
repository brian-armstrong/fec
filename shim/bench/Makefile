# Head-to-head benchmarks: fec (via the C-ABI shim) vs libfec, using libfec's
# own test programs. Only the codec library differs between the two builds, so
# the comparison is apples-to-apples.
#
#   make run        # build and run both the 64-bit and 32-bit tables
#   make run64      # 64-bit only: fec vs libfec portable C (no multilib needed)
#   make run32      # 32-bit only: fec vs libfec's SSE2 asm and portable C
#                   #   (needs gcc multilib and a 32-bit assembler)
#   make clean
#
# Everything is built from source at a known -O2, so a fresh clone reproduces
# the numbers. Each arch builds into its own subdir (out64/, out32/) so the two
# coexist and switching between them never forces a rebuild.
#
# The libfec baseline is built from source into a private copy under
# out<arch>/libfec, using libfec's OWN autoconf configure at our flags. configure
# selects both the codec objects and the cpu-mode detection for the target: the
# 32-bit build gets Karn's SSE2 asm and a cpu_mode that detects it at runtime, the
# 64-bit build gets portable C (libfec has no 64-bit SIMD). So a default no-flag
# run already uses each build's best kernel. The real libfec checkout stays
# untouched and there is no dependence on prebuilt archives of unknown vintage.
#
# LIBFEC points at a libfec checkout (source only is used). SHIM_A is resolved
# per arch from the cargo target dir.

LIBFEC ?= ../../../libfec
TRIALS ?= 2000

CC = gcc

.PHONY: run run64 run32 do64 do32 clean
.DEFAULT_GOAL := run

# --- per-arch dispatch -------------------------------------------------------
# `make run` runs both; the sub-invocations recurse with ARCH set, which selects
# the flags, output dir, and shim path below.

run: run64 run32
run64:
	@$(MAKE) --no-print-directory _run ARCH=64
run32:
	@$(MAKE) --no-print-directory _run ARCH=32
do64:
	@$(MAKE) --no-print-directory _build ARCH=64
do32:
	@$(MAKE) --no-print-directory _build ARCH=32

# --- arch-specific settings (chosen by ARCH, default 64) ---------------------

ARCH ?= 64

ifeq ($(ARCH),32)
  ARCHFLAGS = -m32
  OUT       = out32
  CARGO_TARGET = i686-unknown-linux-gnu
  SHIM_A       = ../../target/$(CARGO_TARGET)/release/libfec.a
  CARGO_TFLAG  = --target $(CARGO_TARGET)
  # libfec's configure selects its SSE2 asm kernels only when the canonical host
  # cpu is i386/i686. On an x86_64 machine -m32 alone does not change that, so we
  # must tell configure the host explicitly to get Karn's asm column.
  CONFIGURE_HOST = --host=i686-pc-linux-gnu
  # The .s kernels assemble through bare `as`, which defaults to 64-bit on this
  # machine and rejects the 32-bit asm. Force a 32-bit assembler.
  AS_FOR_LIBFEC = as --32
else
  ARCHFLAGS = -m64
  OUT       = out64
  SHIM_A    = ../../target/release/libfec.a
  CARGO_TFLAG =
  CONFIGURE_HOST =
  AS_FOR_LIBFEC = as
endif

CFLAGS = -O2 $(ARCHFLAGS) -I$(LIBFEC)
LDLIBS = -lm

# Absolute path to the libfec source, so the private-copy build (which runs from
# a different directory) resolves it regardless of ARCH.
LIBFEC_ABS := $(abspath $(LIBFEC))

# The private libfec copy and the baseline archive built inside it.
LIBFEC_SRC = $(OUT)/libfec
LIBFEC_A   = $(LIBFEC_SRC)/libfec.a

VTESTS      = 27 29 39 615
LIBFEC_BINS = $(addprefix $(OUT)/vtest,$(addsuffix _libfec,$(VTESTS))) $(OUT)/rs_speedtest_libfec $(OUT)/rs_errtest_libfec
FEC_BINS    = $(addprefix $(OUT)/vtest,$(addsuffix _fec,$(VTESTS)))    $(OUT)/rs_speedtest_fec    $(OUT)/rs_errtest_fec

# libfec infrastructure the fec test programs need but the shim does not provide:
# the parity table (Partab from fec.o), the cpu-mode symbol (Cpu_mode from
# cpu_mode_unknown.o), and the noise helper (addnoise from sim.c). These are NOT
# codec symbols, so they never collide with the shim. fec.o and
# cpu_mode_unknown.o are already built as a side effect of the libfec.a build in
# the copy, at the right bitness, so we reuse them rather than recompiling. We
# cannot link the whole libfec.a for them because it also defines the codec
# symbols, which would collide with the shim.
HELPERS = $(LIBFEC)/sim.c $(LIBFEC_SRC)/fec.o $(LIBFEC_SRC)/cpu_mode_unknown.o

_build: $(LIBFEC_BINS) $(FEC_BINS)

$(OUT):
	mkdir -p $(OUT)

# --- private libfec copy, configured and built at our flags ------------------
# libfec is autoconf. A fresh clone ships configure + makefile.in but no
# makefile, and configure is what selects MLIBS: the SSE2 asm kernels for a
# 32-bit x86 host, portable C for 64-bit. So we rsync the sources into
# out<arch>/libfec, run libfec's own configure with our CFLAGS,
# and build libfec.a there. The copy keeps this immune to a source tree that was
# previously built in place, and leaves the real checkout untouched.
$(LIBFEC_A): | $(OUT)
	rsync -a --delete \
	  --exclude='*.o' --exclude='*.a' --exclude='*.so' \
	  --exclude='makefile' --exclude='build*' --exclude='.git' \
	  $(LIBFEC_ABS)/ $(LIBFEC_SRC)/
	cd $(LIBFEC_SRC) && ./configure $(CONFIGURE_HOST) CC=$(CC) "CFLAGS=-O2 $(ARCHFLAGS) -fPIC"
	$(MAKE) -C $(LIBFEC_SRC) "AS=$(AS_FOR_LIBFEC)" libfec.a

# fec.o and cpu_mode_unknown.o fall out of the libfec.a build in the copy, so
# they share its recipe rather than being compiled separately.
$(LIBFEC_SRC)/fec.o $(LIBFEC_SRC)/cpu_mode_unknown.o: $(LIBFEC_A)

# --- Viterbi: baseline links our libfec.a; fec links the shim + libfec helpers.
# Both link libfec's sim.c the same way, so the harness is identical. -----------
$(OUT)/vtest%_libfec: $(LIBFEC)/vtest%.c $(LIBFEC_A) | $(OUT)
	$(CC) $(CFLAGS) $< $(LIBFEC)/sim.c $(LIBFEC_A) $(LDLIBS) -o $@

$(OUT)/vtest%_fec: $(LIBFEC)/vtest%.c $(SHIM_A) $(HELPERS) | $(OUT)
	$(CC) $(CFLAGS) $< $(HELPERS) $(SHIM_A) $(LDLIBS) -lpthread -ldl -o $@

# --- Reed-Solomon speed test ---
# rs_speedtest.c hardcodes 10000 trials, which decodes so fast that the runs land
# in the 0.01s timer's quantization noise. We sed the trial count up into a
# build-local copy so the runs are long enough to time precisely. A second copy
# flips the test's `#if 0` error injection on, so it decodes a block with two
# symbol errors (the full correction path) instead of a clean block.
RS_TRIALS ?= 500000

$(OUT)/rs_speedtest.c: $(LIBFEC)/rs_speedtest.c | $(OUT)
	sed 's/int trials = 10000;/int trials = $(RS_TRIALS);/' $< > $@

$(OUT)/rs_errtest.c: $(LIBFEC)/rs_speedtest.c | $(OUT)
	sed -e 's/int trials = 10000;/int trials = $(RS_TRIALS);/' -e 's/#if 0/#if 1/' $< > $@

$(OUT)/rs_speedtest_libfec: $(OUT)/rs_speedtest.c $(LIBFEC_A) | $(OUT)
	$(CC) $(CFLAGS) $< $(LIBFEC_A) $(LDLIBS) -o $@

$(OUT)/rs_speedtest_fec: $(OUT)/rs_speedtest.c $(SHIM_A) $(HELPERS) | $(OUT)
	$(CC) $(CFLAGS) $< $(HELPERS) $(SHIM_A) $(LDLIBS) -lpthread -ldl -o $@

$(OUT)/rs_errtest_libfec: $(OUT)/rs_errtest.c $(LIBFEC_A) | $(OUT)
	$(CC) $(CFLAGS) $< $(LIBFEC_A) $(LDLIBS) -o $@

$(OUT)/rs_errtest_fec: $(OUT)/rs_errtest.c $(SHIM_A) $(HELPERS) | $(OUT)
	$(CC) $(CFLAGS) $< $(HELPERS) $(SHIM_A) $(LDLIBS) -lpthread -ldl -o $@

# Build the shim static lib for this arch before benchmarking against it. The
# simd feature is nightly-only, so this must use +nightly.
$(SHIM_A):
	cd ../.. && cargo +nightly build -p fec-shim --release --features simd $(CARGO_TFLAG)

# --- run one arch's table ----------------------------------------------------
# vtest calls find_cpu_mode() before parsing flags, so a no-flag run uses the
# kernel the build detects: the 32-bit build detects and runs Karn's SSE2 asm,
# the 64-bit build has only portable C. Explicit flags override: -t forces SSE2,
# -p forces portable. So on 32-bit we show both -p (portable) and the default
# (SSE2) to expose the asm-vs-C gap; on 64-bit the default is already portable.

_run: _build
	@echo "########## $(ARCH)-bit ##########"
	@for v in $(VTESTS); do \
	  echo "=== rate/k=$$v ==="; \
	  if [ "$(ARCH)" = "32" ]; then \
	    printf "  libfec SSE2 (asm):  "; ./$(OUT)/vtest$${v}_libfec    -n $(TRIALS) 2>&1 | grep "Execution time"; \
	    printf "  libfec port (C):    "; ./$(OUT)/vtest$${v}_libfec -p -n $(TRIALS) 2>&1 | grep "Execution time"; \
	  else \
	    printf "  libfec port (C):    "; ./$(OUT)/vtest$${v}_libfec    -n $(TRIALS) 2>&1 | grep "Execution time"; \
	  fi; \
	  printf "  fec         ($(ARCH)b): "; ./$(OUT)/vtest$${v}_fec    -n $(TRIALS) 2>&1 | grep "Execution time"; \
	done
	@echo "=== Reed-Solomon (255,223), general + CCSDS decoders ==="
	@echo "--- clean block ---"
	@printf "  libfec ($(ARCH)b): "; ./$(OUT)/rs_speedtest_libfec 2>&1 | grep "decoder speed" | tr '\n' ' '; echo
	@printf "  fec    ($(ARCH)b): "; ./$(OUT)/rs_speedtest_fec    2>&1 | grep "decoder speed" | tr '\n' ' '; echo
	@echo "--- 2 symbol errors ---"
	@printf "  libfec ($(ARCH)b): "; ./$(OUT)/rs_errtest_libfec 2>&1 | grep "decoder speed" | tr '\n' ' '; echo
	@printf "  fec    ($(ARCH)b): "; ./$(OUT)/rs_errtest_fec    2>&1 | grep "decoder speed" | tr '\n' ' '; echo

clean:
	rm -rf out64 out32
