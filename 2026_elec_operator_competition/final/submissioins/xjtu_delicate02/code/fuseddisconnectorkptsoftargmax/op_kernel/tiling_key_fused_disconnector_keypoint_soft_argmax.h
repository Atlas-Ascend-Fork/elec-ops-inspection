#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(FusedDisconnectorKeypointSoftArgmax,
    ASCENDC_TPL_DATATYPE_DECL(DT_FEAT, C_DT_FLOAT),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_FEAT, C_DT_FLOAT),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
);
