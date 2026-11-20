using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace SlicerDSpace.WslBridge
{
    internal sealed class BridgeConfig
    {
        internal string Distribution = "Ubuntu";
        internal string Target = "";
        internal string Supervisor = "";
        internal string SupervisorPython = "";
        internal readonly List<string> PrefixArguments = new List<string>();
        internal readonly List<KeyValuePair<string, string>> Mappings =
            new List<KeyValuePair<string, string>>();

        internal static BridgeConfig Load(string executablePath)
        {
            var configPath = executablePath + ".conf";
            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException("WSL bridge configuration is missing", configPath);
            }

            var result = new BridgeConfig();
            foreach (var rawLine in File.ReadAllLines(configPath, Encoding.UTF8))
            {
                var line = rawLine.Trim();
                if (line.Length == 0 || line.StartsWith("#", StringComparison.Ordinal))
                {
                    continue;
                }

                var separator = line.IndexOf('=');
                if (separator <= 0)
                {
                    throw new InvalidDataException("Invalid WSL bridge setting: " + line);
                }

                var key = line.Substring(0, separator).Trim();
                var value = line.Substring(separator + 1);
                if (key.Equals("distribution", StringComparison.OrdinalIgnoreCase))
                {
                    result.Distribution = value;
                }
                else if (key.Equals("target", StringComparison.OrdinalIgnoreCase))
                {
                    result.Target = value;
                }
                else if (key.Equals("supervisor", StringComparison.OrdinalIgnoreCase))
                {
                    result.Supervisor = value;
                }
                else if (key.Equals("supervisor_python", StringComparison.OrdinalIgnoreCase))
                {
                    result.SupervisorPython = value;
                }
                else if (key.Equals("map", StringComparison.OrdinalIgnoreCase))
                {
                    var mappingSeparator = value.IndexOf('|');
                    if (mappingSeparator <= 0 || mappingSeparator == value.Length - 1)
                    {
                        throw new InvalidDataException("Invalid WSL bridge path mapping: " + value);
                    }
                    result.Mappings.Add(
                        new KeyValuePair<string, string>(
                            value.Substring(0, mappingSeparator),
                            value.Substring(mappingSeparator + 1)));
                }
                else if (key.Equals("arg", StringComparison.OrdinalIgnoreCase))
                {
                    result.PrefixArguments.Add(value);
                }
            }

            if (String.IsNullOrWhiteSpace(result.Distribution))
            {
                throw new InvalidDataException("WSL bridge distribution is empty");
            }
            if (String.IsNullOrWhiteSpace(result.Target) || !result.Target.StartsWith("/", StringComparison.Ordinal))
            {
                throw new InvalidDataException("WSL bridge target must be an absolute Linux path");
            }
            if (result.Supervisor.Length > 0 &&
                (!result.Supervisor.StartsWith("/", StringComparison.Ordinal) ||
                 !result.SupervisorPython.StartsWith("/", StringComparison.Ordinal)))
            {
                throw new InvalidDataException("WSL supervisor and Python must have absolute Linux paths");
            }
            return result;
        }
    }

    internal static class Program
    {
        private static readonly Regex DrivePath = new Regex(
            @"^[A-Za-z]:[\\/]", RegexOptions.CultureInvariant | RegexOptions.Compiled);

        private static string QuoteWindowsArgument(string value)
        {
            if (value.Length > 0 && value.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '"' }) < 0)
            {
                return value;
            }

            var result = new StringBuilder();
            result.Append('"');
            var backslashes = 0;
            foreach (var character in value)
            {
                if (character == '\\')
                {
                    backslashes += 1;
                    continue;
                }
                if (character == '"')
                {
                    result.Append('\\', backslashes * 2 + 1);
                    result.Append('"');
                    backslashes = 0;
                    continue;
                }
                result.Append('\\', backslashes);
                backslashes = 0;
                result.Append(character);
            }
            result.Append('\\', backslashes * 2);
            result.Append('"');
            return result.ToString();
        }

        private static string BuildArguments(IEnumerable<string> values)
        {
            var quoted = new List<string>();
            foreach (var value in values)
            {
                quoted.Add(QuoteWindowsArgument(value));
            }
            return String.Join(" ", quoted.ToArray());
        }

        private static string NormalizeWindowsPath(string value)
        {
            return value.Replace('/', '\\').TrimEnd('\\');
        }

        private static string TryConfiguredMapping(string value, BridgeConfig config)
        {
            var normalized = NormalizeWindowsPath(value);
            foreach (var mapping in config.Mappings)
            {
                if (String.Equals(
                    normalized,
                    NormalizeWindowsPath(mapping.Key),
                    StringComparison.OrdinalIgnoreCase))
                {
                    return mapping.Value;
                }
            }
            return null;
        }

        private static string TryWslUncPath(string value, string distribution)
        {
            var normalized = value.Replace('/', '\\');
            var prefixes = new[]
            {
                @"\\wsl.localhost\" + distribution + @"\",
                @"\\wsl$\" + distribution + @"\",
            };
            foreach (var prefix in prefixes)
            {
                if (normalized.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    return "/" + normalized.Substring(prefix.Length).Replace('\\', '/');
                }
            }
            return null;
        }

        private const int WslPathTimeoutMilliseconds = 30000;

        private static string ConvertDrivePath(string value, BridgeConfig config)
        {
            var portablePath = value.Replace('\\', '/');
            var startInfo = new ProcessStartInfo
            {
                FileName = "wsl.exe",
                Arguments = BuildArguments(new[]
                {
                    "--distribution", config.Distribution, "--exec", "wslpath", "-a", "-u", portablePath,
                }),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                // X2: wslpath prints UTF-8. The default OEM code page turns a non-ASCII user or temp path into mojibake.
                StandardOutputEncoding = new UTF8Encoding(false),
                StandardErrorEncoding = new UTF8Encoding(false),
                CreateNoWindow = true,
            };
            using (var process = Process.Start(startInfo))
            {
                var stdoutBuilder = new StringBuilder();
                var stderrBuilder = new StringBuilder();
                process.OutputDataReceived += (sender, e) =>
                {
                    if (e.Data != null) { lock (stdoutBuilder) { stdoutBuilder.AppendLine(e.Data); } }
                };
                process.ErrorDataReceived += (sender, e) =>
                {
                    if (e.Data != null) { lock (stderrBuilder) { stderrBuilder.AppendLine(e.Data); } }
                };
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                if (!process.WaitForExit(WslPathTimeoutMilliseconds))
                {
                    try { process.Kill(); } catch (Exception) { }
                    throw new InvalidOperationException(
                        "wslpath did not finish within 30 s for '" + value + "'. Check that WSL distribution '"
                        + config.Distribution + "' starts (run: wsl.exe --distribution " + config.Distribution
                        + " --exec true), then retry.");
                }
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    throw new InvalidOperationException(
                        "wslpath failed for '" + value + "': " + stderrBuilder.ToString().Trim());
                }
                return stdoutBuilder.ToString().TrimEnd('\r', '\n');
            }
        }

        private static string ConvertArgument(string value, BridgeConfig config)
        {
            var configured = TryConfiguredMapping(value, config);
            if (configured != null)
            {
                return configured;
            }
            var uncPath = TryWslUncPath(value, config.Distribution);
            if (uncPath != null)
            {
                return uncPath;
            }
            return DrivePath.IsMatch(value) ? ConvertDrivePath(value, config) : value;
        }

        private static int Main(string[] args)
        {
            string heartbeat = null;
            Timer heartbeatTimer = null;
            try
            {
                var executablePath = Process.GetCurrentProcess().MainModule.FileName;
                var config = BridgeConfig.Load(executablePath);
                var command = new List<string>
                {
                    "--distribution", config.Distribution, "--exec",
                };
                if (config.Supervisor.Length > 0)
                {
                    heartbeat = Path.GetTempFileName();
                    var localHeartbeat = heartbeat;
                    heartbeatTimer = new Timer(_ =>
                    {
                        try { File.SetLastWriteTimeUtc(localHeartbeat, DateTime.UtcNow); }
                        catch (IOException) { /* The supervisor fails closed on expired heartbeat. */ }
                        catch (UnauthorizedAccessException) { }
                    }, null, 0, 1000);
                    command.Add(config.SupervisorPython);
                    command.Add(config.Supervisor);
                    command.Add("--heartbeat");
                    command.Add(ConvertArgument(heartbeat, config));
                    command.Add("--");
                }
                // Keep scheduler/user GPU restrictions without leaking the
                // Windows PATH or Slicer Python environment into Linux.
                var cudaVisibility = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
                if (cudaVisibility != null)
                {
                    command.Add("/usr/bin/env");
                    command.Add("CUDA_VISIBLE_DEVICES=" + cudaVisibility);
                }
                command.Add(config.Target);
                command.AddRange(config.PrefixArguments);
                foreach (var argument in args)
                {
                    command.Add(ConvertArgument(argument, config));
                }

                using (var process = Process.Start(new ProcessStartInfo
                {
                    FileName = "wsl.exe",
                    Arguments = BuildArguments(command),
                    UseShellExecute = false,
                }))
                {
                    process.WaitForExit();
                    return process.ExitCode;
                }
            }
            catch (Exception error)
            {
                Console.Error.WriteLine("SlicerDSpace WSL bridge failed: " + error.Message);
                return 125;
            }
            finally
            {
                if (heartbeatTimer != null) { heartbeatTimer.Dispose(); }
                if (heartbeat != null)
                {
                    try { File.Delete(heartbeat); }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                }
            }
        }
    }
}
