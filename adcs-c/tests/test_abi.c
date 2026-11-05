/* Placeholder test until the Unity framework is added with the first algorithms. */
#include <stdio.h>

#include "adcs.h"

int main(void)
{
    if (adcs_abi_version() != ADCS_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch: library %u, header %u\n",
                (unsigned)adcs_abi_version(), (unsigned)ADCS_ABI_VERSION);
        return 1;
    }
    return 0;
}
