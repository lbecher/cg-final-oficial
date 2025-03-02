use std::fmt::Display;
use std::str::FromStr;

pub fn parse_input<T>(prefix: &str, value: &mut T, string: &mut String)
where
    T: FromStr + Display + Copy,
    T::Err: std::fmt::Display,
{
    if string.starts_with(prefix) {
        if let Some(num_str) = string.strip_prefix(prefix) {
            if let Ok(parsed_value) = num_str.trim().parse::<T>() {
                *value = parsed_value;
                *string = format!("{prefix} {}", parsed_value);
                return;
            }
        }
    } else {
        if let Ok(parsed_value) = string.trim().parse::<T>() {
            *value = parsed_value;
            *string = format!("{prefix} {}", parsed_value);
            return;
        }
    }
    *string = format!("Inválido!");
}