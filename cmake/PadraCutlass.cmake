include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/PadraSourceLock.cmake")

function(padra_add_cutlass)
  set(revision "db1c288993354c88e551c40c19a8fb93a774a241")
  set(manifest "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cutlass-source-lock.json")
  file(SHA256 "${manifest}" manifest_hash)
  if(NOT manifest_hash STREQUAL "c1d04d1dd2c64156a827bc31ce267a3a9023291c8ef7ffbc35c9488c32075557")
    message(FATAL_ERROR "CUTLASS source lock hash mismatch")
  endif()
  set(PADRA_CUTLASS_ROOT "" CACHE PATH "Optional pinned CUTLASS source tree or prepared header cache; empty downloads automatically")
  if(PADRA_CUTLASS_ROOT)
    get_filename_component(source_dir "${PADRA_CUTLASS_ROOT}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
    # Preserve the old prepared-cache identity check when a lock is present.
    if(EXISTS "${source_dir}/SOURCE_LOCK.json")
      file(SHA256 "${source_dir}/SOURCE_LOCK.json" cache_lock_hash)
      if(NOT cache_lock_hash STREQUAL manifest_hash)
        message(FATAL_ERROR "CUTLASS prepared cache source lock hash mismatch")
      endif()
    endif()
  else()
    include(FetchContent)
    FetchContent_Declare(padra_cutlass_source
      URL "https://codeload.github.com/NVIDIA/cutlass/tar.gz/${revision}"
      URL_HASH SHA256=e84e148a587cbe62ab29118389fd842fa77af5c58c1f6a940d6297669c8c3a6f
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      TLS_VERIFY TRUE
      # Headers only: do not configure CUTLASS's examples, tests or tools.
      SOURCE_SUBDIR include)
    FetchContent_MakeAvailable(padra_cutlass_source)
    set(source_dir "${padra_cutlass_source_SOURCE_DIR}")
  endif()
  padra_verify_sources("${source_dir}" "${manifest}" "CUTLASS")
  add_library(padra_cutlass INTERFACE)
  add_library(Padra::CUTLASS ALIAS padra_cutlass)
  target_include_directories(padra_cutlass SYSTEM INTERFACE
    "${source_dir}/include" "${source_dir}/tools/util/include")
  message(STATUS "CUTLASS ${revision}: verified headers in ${source_dir}")
endfunction()
