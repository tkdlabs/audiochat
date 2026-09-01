use cpal::traits::{DeviceTrait, HostTrait};

fn main() {
    let host = cpal::default_host();
    match host.default_input_device() {
        Some(d) => match d.default_input_config() {
            Ok(c) => {
                println!(
                    "input: fmt={:?} rate={} channels={}",
                    c.sample_format(),
                    c.sample_rate(),
                    c.channels()
                );
            }
            Err(e) => println!("input config error: {e}"),
        },
        None => println!("NO default input device"),
    }
    match host.default_output_device() {
        Some(d) => match d.default_output_config() {
            Ok(c) => {
                println!(
                    "output: fmt={:?} rate={} channels={}",
                    c.sample_format(),
                    c.sample_rate(),
                    c.channels()
                );
            }
            Err(e) => println!("output config error: {e}"),
        },
        None => println!("NO default output device"),
    }
}
