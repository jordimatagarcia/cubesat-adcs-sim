/*
 * adcs.h: public interface of the ADCS library.
 *
 * This header is the contract shared by the C and Rust implementations. Both libraries
 * export exactly these symbols, so the simulator can load either one.
 */
#ifndef ADCS_H
#define ADCS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of this interface. Incremented on any change to the exported structs or functions. */
#define ADCS_ABI_VERSION 1u

/* Returns ADCS_ABI_VERSION as compiled into the library. */
uint32_t adcs_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ADCS_H */
