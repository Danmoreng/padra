include_guard(GLOBAL)

# The manifest lives in this repository, so local/offline sources receive the
# same checks as downloaded sources. No Git checkout or Python is required.
function(padra_verify_sources root manifest dependency)
  file(READ "${manifest}" source_lock)
  string(JSON file_count LENGTH "${source_lock}" sha256)
  math(EXPR last_file "${file_count}-1")
  set(verified_files "${manifest}")
  foreach(index RANGE ${last_file})
    string(JSON relative MEMBER "${source_lock}" sha256 ${index})
    string(JSON expected GET "${source_lock}" sha256 "${relative}")
    if(NOT EXISTS "${root}/${relative}")
      message(FATAL_ERROR "${dependency} source missing: ${root}/${relative}")
    endif()
    file(SHA256 "${root}/${relative}" actual)
    if(NOT actual STREQUAL expected)
      message(FATAL_ERROR "${dependency} source hash mismatch: ${root}/${relative}")
    endif()
    list(APPEND verified_files "${root}/${relative}")
  endforeach()
  # Recheck on the next normal build if pinned sources or their lock change.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${verified_files})
endfunction()
