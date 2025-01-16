pub fn parse_input(prefix: &str, value: &mut f32, string: &mut String) {
    if string.starts_with(prefix) {
        if let Some(num_str) = string.strip_prefix(prefix) {
            if let Ok(parsed_value) = num_str.trim().parse::<f32>() {
                *value = parsed_value;
                *string = format!("{prefix} {parsed_value}");
                return;
            }
        }
    } else {
        if let Ok(parsed_value) = string.trim().parse::<f32>() {
            *value = parsed_value;
            *string = format!("{prefix} {parsed_value}");
            return;
        }
    }
    *string = format!("Inválido!");
}