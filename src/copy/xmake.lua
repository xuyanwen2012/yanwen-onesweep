
add_requires("spdlog")

target("demo")
    set_kind("binary")
    add_includedirs("../../include")
    add_files("./*.cu")
    add_files("../init.cu")
    add_packages("spdlog")
    
    -- generate SASS code for SM architecture of current host
    add_cugencodes("native")

    -- generate PTX code for the virtual architecture to guarantee compatibility
    add_cugencodes("compute_35")

