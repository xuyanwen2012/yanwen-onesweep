add_rules("mode.debug", "mode.release")

set_languages("c++17")
set_warnings("all")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
end

target("yanwen-onesweep")
    set_kind("binary")
    add_includedirs("include")
    add_headerfiles("include/*", "include/**/*")
    add_files("src/*.cu")
    add_cugencodes("native")

-- includes("src/copy")
