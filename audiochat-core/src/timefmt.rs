//! Dependency-free UTC wall-clock formatting for prompt side-information.
//!
//! Timestamps are rendered in UTC (the standard library exposes no local
//! offset), which is enough for an LLM to reason about date, time of day, and
//! elapsed time between messages.

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

/// Build the bracketed side-information line for a turn. `gap` is the time since
/// the previous exchange, or `None` for the first message this session.
pub fn context_line(now_secs: u64, gap: Option<Duration>) -> String {
    match gap {
        Some(gap) => format!(
            "[context: {}; {} since the previous exchange]",
            format_timestamp(now_secs),
            format_duration(gap)
        ),
        None => format!(
            "[context: {}; start of session]",
            format_timestamp(now_secs)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_epoch() {
        assert_eq!(format_timestamp(0), "1970-01-01 00:00:00 UTC (Thursday)");
    }

    #[test]
    fn formats_known_instant() {
        // 2021-01-01 00:00:00 UTC was a Friday.
        assert_eq!(
            format_timestamp(1_609_459_200),
            "2021-01-01 00:00:00 UTC (Friday)"
        );
    }

    #[test]
    fn handles_leap_day() {
        // 2024-02-29 12:00:00 UTC was a Thursday.
        assert_eq!(
            format_timestamp(1_709_208_000),
            "2024-02-29 12:00:00 UTC (Thursday)"
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
        assert_eq!(
            first,
            "[context: 1970-01-01 00:00:00 UTC (Thursday); start of session]"
        );

        let later = context_line(0, Some(Duration::from_secs(180)));
        assert!(later.contains("3 minutes since the previous exchange"));
    }
}
