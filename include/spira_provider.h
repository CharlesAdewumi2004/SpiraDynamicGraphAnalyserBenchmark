#pragma once

#include "provider.h"
#include <cstddef>
#include <memory>

std::unique_ptr<SpMVProvider> make_spira_provider(std::size_t n_threads);
