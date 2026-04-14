#pragma once

#include "provider.h"
#include <memory>

std::unique_ptr<SpMVProvider> make_csr_reference();
