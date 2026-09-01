//! Diagnostics: list available audio input/output devices and their configs.

use cpal::traits::{DeviceTrait, HostTrait};

fn main() {
    let host = cpal::default_host();

    println!("== Input devices ==");
    match host.input_devices() {
        Ok(devices) => {
            for device in devices {
                let name = device.to_string();
                match device.default_input_config() {
                    Ok(c) => println!(
                        "  {name}: fmt={:?} rate={} channels={}",
                        c.sample_format(),
                        c.sample_rate(),
                        c.channels()
                    ),
                    Err(e) => println!("  {name}: config error: {e}"),
                }
            }
        }
        Err(e) => println!("  failed to enumerate: {e}"),
    }

    if host.default_input_device().is_some() {
        println!("  [default input] -> match --device against names above");
    }

    println!("== Default output ==");
    if let Some(device) = host.default_output_device() {
        let name = device.to_string();
        match device.default_output_config() {
            Ok(c) => println!(
                "  {name}: fmt={:?} rate={} channels={}",
                c.sample_format(),
                c.sample_rate(),
                c.channels()
            ),
            Err(e) => println!("  {name}: config error: {e}"),
        }
    } else {
        println!("  no default output device");
    }
}
