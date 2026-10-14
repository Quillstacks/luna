# `luna.io` Reference

The `luna.io` package handles data fetching, PDS index lookups, NAC decoding, and SPICE photogrammetric coordinate projections.

---

## SPICE Projection (`luna.io.spice_project`)

::: luna.io.spice_project
    options:
      members:
        - ground_to_image
        - image_to_ground
        - lonlat_to_pixel_spice
        - furnish_kernels
        - furnish_kernels_for_date
        - ensure_kernels_for_label
        - MissingKernelError

---

## Digital Elevation Model (`luna.io.dem`)

::: luna.io.dem
    options:
      members:
        - LunarDEM
        - fetch_dem_for_region
        - get_sldem2015_tile_name
        - default_dem

---

## Linear Projection (`luna.io.projection`)

::: luna.io.projection
    options:
      members:
        - LinearProjection
        - lonlat_to_pixel
        - pixel_to_lonlat
        - get_image_of_roi

---

## PDS Index Resolver (`luna.io.pds_index`)

::: luna.io.pds_index
    options:
      members:
        - PDSIndex

---

## NAC Reader (`luna.io.nac_reader`)

::: luna.io.nac_reader
    options:
      members:
        - NACImage
        - read_nac

---

## PDS Fetcher (`luna.io.pds_fetch`)

::: luna.io.pds_fetch
    options:
      members:
        - fetch_nac
        - get_pds_session
        - BandwidthLimiter
        - DEFAULT_USER_AGENT

---

## Kernel Fetcher (`luna.io.kernel_fetch`)

::: luna.io.kernel_fetch
    options:
      members:
        - ensure_kernels_for_date

---

## Mosaicking & Coverage Solver (`luna.io.coverage`)

::: luna.io.coverage
    options:
      members:
        - select_coverage_nacs
