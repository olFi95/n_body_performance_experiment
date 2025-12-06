use wgsl_bindgen::{WgslTypeSerializeStrategy, WgslBindgenOptionBuilder, GlamWgslTypeMap, Regex};
fn main() -> anyhow::Result<()> {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not defined by Cargo");
    let generated = WgslBindgenOptionBuilder::default()
        .workspace_root("src/nbody/shaders")
        .add_entry_point("src/nbody/shaders/nbody.wgsl")
        .custom_padding_field_regexps(vec![Regex::new("^padding[0-9]+$").unwrap()])
        .skip_hash_check(false)
        .serialization_strategy(WgslTypeSerializeStrategy::Bytemuck)
        .type_map(GlamWgslTypeMap)
        .derive_serde(false)
        .build()?
        .generate_string()?;
    let sanitized = generated.replace("#![allow", "#[allow");
    // wgsl-bindgen emits an inner attribute (`#![allow(...)]`) at the top of the
    // generated file. When this file is included inside a Rust module via `include!()`,
    // inner attributes are not permitted and cause a compiler error.
    // We sanitize the output by converting the inner attribute to an outer one (`#[allow(...)]`),
    // which *is* valid inside a module.
    std::fs::write(format!("{}/shaders_types.rs", out_dir), sanitized)?;
    Ok(())
}