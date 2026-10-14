//! Dependency-light wall-clock formatting for prompt side-information.
//!
//! Timestamps are rendered in the machine's local time where possible, with the
//! UTC offset and (roughly) the IANA time zone, so a model can reason about the
//! date, time of day, elapsed time between messages, and where the user is.
//!
//! The local offset comes from the C library's thread-safe `localtime_r` (Unix
//! only); elsewhere the formatter falls back to UTC.

use std::sync::OnceLock;
use std::time::Duration;

/// Split Unix seconds into a civil date and time (UTC).
fn civil_from_unix(secs: u64) -> (i64, u32, u32, u64, u64, u64, usize) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = rem / 3_600;
    let minute = (rem % 3_600) / 60;
    let second = rem % 60;
    // 1970-01-01 was a Thursday; with 0 = Sunday that is index 4.
    let weekday = (days + 4).rem_euclid(7) as usize;
    (year, month, day, hour, minute, second, weekday)
}

/// Days since the Unix epoch to a `(year, month, day)` civil date, using
/// Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

const WEEKDAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];

/// Format Unix seconds as `2026-10-14 15:32:07 UTC (Wednesday)`.
pub fn format_timestamp(secs: u64) -> String {
    let (year, month, day, hour, minute, second, weekday) = civil_from_unix(secs);
    format!(
        "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC ({})",
        WEEKDAYS[weekday]
    )
}

/// Format a local timestamp from Unix seconds and a UTC offset in seconds, e.g.
/// `2026-10-14 17:32:07 +02:00 (Europe/Berlin, Wednesday)`.
pub fn format_local_timestamp(secs: u64, offset_secs: i64, zone: Option<&str>) -> String {
    let shifted = (secs as i64 + offset_secs).max(0) as u64;
    let (year, month, day, hour, minute, second, weekday) = civil_from_unix(shifted);
    let offset = format_offset(offset_secs);
    let weekday = WEEKDAYS[weekday];
    match zone {
        Some(zone) => format!(
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} {offset} ({zone}, {weekday})"
        ),
        None => format!(
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} {offset} ({weekday})"
        ),
    }
}

/// Format a UTC offset in seconds as `+HH:MM` / `-HH:MM`.
fn format_offset(offset_secs: i64) -> String {
    let sign = if offset_secs < 0 { '-' } else { '+' };
    let abs = offset_secs.unsigned_abs();
    format!("{sign}{:02}:{:02}", abs / 3_600, (abs % 3_600) / 60)
}

/// The machine's UTC offset (seconds) at `now_secs`, or `None` if unavailable.
#[cfg(unix)]
fn local_offset_secs(now_secs: u64) -> Option<i64> {
    let time = now_secs as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let ok = unsafe { !libc::localtime_r(&time, &mut tm).is_null() };
    ok.then_some(tm.tm_gmtoff as i64)
}

#[cfg(not(unix))]
fn local_offset_secs(_now_secs: u64) -> Option<i64> {
    None
}

/// A rough IANA time-zone name for the machine, e.g. `Europe/Berlin`, from the
/// `TZ` environment variable or the `/etc/localtime` symlink. Cached.
fn local_zone_name() -> Option<String> {
    static ZONE: OnceLock<Option<String>> = OnceLock::new();
    ZONE.get_or_init(|| {
        if let Ok(tz) = std::env::var("TZ") {
            let tz = tz.trim().trim_start_matches(':');
            if !tz.is_empty() {
                return Some(tz.to_string());
            }
        }
        #[cfg(unix)]
        if let Ok(link) = std::fs::read_link("/etc/localtime") {
            let link = link.to_string_lossy();
            if let Some(idx) = link.find("zoneinfo/") {
                let zone = link[idx + "zoneinfo/".len()..].trim();
                if !zone.is_empty() {
                    return Some(zone.to_string());
                }
            }
        }
        None
    })
    .clone()
}

/// A short human phrase for a duration, e.g. `45 seconds`, `3 minutes`,
/// `2 hours 5 minutes`, `2 days 3 hours`.
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        return plural(secs, "second");
    }
    if secs < 3_600 {
        return plural(secs / 60, "minute");
    }
    if secs < 86_400 {
        let hours = secs / 3_600;
        let minutes = (secs % 3_600) / 60;
        return if minutes == 0 {
            plural(hours, "hour")
        } else {
            format!("{} {minutes} minutes", plural(hours, "hour"))
        };
    }
    let days = secs / 86_400;
    let hours = (secs % 86_400) / 3_600;
    if hours == 0 {
        plural(days, "day")
    } else {
        format!("{} {hours} hours", plural(days, "day"))
    }
}

fn plural(n: u64, unit: &str) -> String {
    if n == 1 {
        format!("1 {unit}")
    } else {
        format!("{n} {unit}s")
    }
}

/// Build the bracketed side-information line for a turn, in local time when the
/// offset is known (falling back to UTC). `gap` is the time since the previous
/// exchange, or `None` for the first message this session.
pub fn context_line(now_secs: u64, gap: Option<Duration>) -> String {
    let timestamp = match local_offset_secs(now_secs) {
        Some(offset) => format_local_timestamp(now_secs, offset, local_zone_name().as_deref()),
        None => format_timestamp(now_secs),
    };
    match gap {
        Some(gap) => format!(
            "[context: {timestamp}; {} since the previous exchange]",
            format_duration(gap)
        ),
        None => format!("[context: {timestamp}; start of session]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_epoch_utc() {
        assert_eq!(format_timestamp(0), "1970-01-01 00:00:00 UTC (Thursday)");
    }

    #[test]
    fn formats_known_instant_utc() {
        // 2021-01-01 00:00:00 UTC was a Friday.
        assert_eq!(
            format_timestamp(1_609_459_200),
            "2021-01-01 00:00:00 UTC (Friday)"
        );
    }

    #[test]
    fn handles_leap_day_utc() {
        // 2024-02-29 12:00:00 UTC was a Thursday.
        assert_eq!(
            format_timestamp(1_709_208_000),
            "2024-02-29 12:00:00 UTC (Thursday)"
        );
    }

    #[test]
    fn formats_local_with_positive_offset_and_zone() {
        assert_eq!(
            format_local_timestamp(0, 7_200, Some("Europe/Berlin")),
            "1970-01-01 02:00:00 +02:00 (Europe/Berlin, Thursday)"
        );
    }

    #[test]
    fn formats_local_with_negative_offset() {
        // 2021-01-01 00:00 UTC minus 5 hours is 2020-12-31 19:00, a Thursday.
        assert_eq!(
            format_local_timestamp(1_609_459_200, -18_000, None),
            "2020-12-31 19:00:00 -05:00 (Thursday)"
        );
    }

    #[test]
    fn duration_phrases() {
        assert_eq!(format_duration(Duration::from_secs(1)), "1 second");
        assert_eq!(format_duration(Duration::from_secs(45)), "45 seconds");
        assert_eq!(format_duration(Duration::from_secs(60)), "1 minute");
        assert_eq!(format_duration(Duration::from_secs(180)), "3 minutes");
        assert_eq!(format_duration(Duration::from_secs(3_600)), "1 hour");
        assert_eq!(
            format_duration(Duration::from_secs(7_500)),
            "2 hours 5 minutes"
        );
        assert_eq!(format_duration(Duration::from_secs(172_800)), "2 days");
        assert_eq!(
            format_duration(Duration::from_secs(183_600)),
            "2 days 3 hours"
        );
    }

    #[test]
    fn context_line_variants() {
        let first = context_line(0, None);
        assert!(first.starts_with("[context: "));
        assert!(first.ends_with("start of session]"));

        let later = context_line(0, Some(Duration::from_secs(180)));
        assert!(later.contains("3 minutes since the previous exchange"));
    }
}
