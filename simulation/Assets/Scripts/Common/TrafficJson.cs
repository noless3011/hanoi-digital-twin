using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

/// <summary>Small strict JSON codec for the traffic wire contract; no Unity package dependency.</summary>
public static class TrafficJson
{
    public static Dictionary<string, object> Obj(params object[] pairs)
    {
        var result = new Dictionary<string, object>();
        for (int i = 0; i < pairs.Length; i += 2) result.Add((string)pairs[i], pairs[i + 1]);
        return result;
    }
    public static float[] Vec(Vector3 value) => new[] { value.x, value.y, value.z };
    public static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    public static Dictionary<string, object> Object(object value)
        => value as Dictionary<string, object> ?? throw new FormatException("Expected JSON object.");
    public static List<object> List(object value)
        => value as List<object> ?? throw new FormatException("Expected JSON array.");
    public static object Get(Dictionary<string, object> value, string key)
        => value.TryGetValue(key, out object result) ? result : throw new FormatException("Missing " + key + ".");
    public static string String(Dictionary<string, object> value, string key)
        => Get(value, key) as string ?? throw new FormatException("Expected string " + key + ".");
    public static double Number(Dictionary<string, object> value, string key)
    {
        object raw = Get(value, key);
        if (!(raw is double) && !(raw is long) && !(raw is float) && !(raw is int))
            throw new FormatException("Expected number " + key + ".");
        double number = Convert.ToDouble(raw, CultureInfo.InvariantCulture);
        if (!Finite(number)) throw new FormatException("Non-finite " + key + ".");
        return number;
    }
    public static long Integer(Dictionary<string, object> value, string key)
    {
        // JSON 1.0 and true must not pass integer identity/sequence validation.
        object raw = Get(value, key);
        if (raw is long integer) return integer;
        if (raw is int small) return small;
        throw new FormatException("Expected integer " + key + ".");
    }
    public static string Encode(object value)
    {
        var builder = new StringBuilder(); Write(builder, value); return builder.ToString();
    }
    private static void Write(StringBuilder output, object value)
    {
        if (value == null) { output.Append("null"); return; }
        if (value is string text) { Quote(output, text); return; }
        if (value is bool boolean) { output.Append(boolean ? "true" : "false"); return; }
        if (value is float single)
        {
            WriteSingle(output, single); return;
        }
        if (value is double real)
        {
            if (!Finite(real)) throw new FormatException("Cannot encode a non-finite number.");
            output.Append(real.ToString("R", CultureInfo.InvariantCulture)); return;
        }
        // Traffic snapshots contain many small dictionaries and three-float vectors.
        // Typed iteration avoids boxing every dictionary entry and vector component.
        if (value is Dictionary<string, object> fields)
        {
            output.Append('{'); bool first = true;
            foreach (var item in fields)
            {
                if (!first) output.Append(','); first = false;
                Quote(output, item.Key); output.Append(':'); Write(output, item.Value);
            }
            output.Append('}'); return;
        }
        if (value is float[] vector)
        {
            output.Append('[');
            for (int i = 0; i < vector.Length; i++)
            {
                if (i > 0) output.Append(',');
                WriteSingle(output, vector[i]);
            }
            output.Append(']'); return;
        }
        if (value is IDictionary dictionary)
        {
            output.Append('{'); bool first = true;
            foreach (DictionaryEntry item in dictionary)
            {
                if (!first) output.Append(','); first = false;
                Quote(output, (string)item.Key); output.Append(':'); Write(output, item.Value);
            }
            output.Append('}'); return;
        }
        if (value is IEnumerable array)
        {
            output.Append('['); bool first = true;
            foreach (object item in array) { if (!first) output.Append(','); first = false; Write(output, item); }
            output.Append(']'); return;
        }
        if (value is IConvertible)
        {
            double number = Convert.ToDouble(value, CultureInfo.InvariantCulture);
            if (!Finite(number)) throw new FormatException("Cannot encode a non-finite number.");
            output.Append(Convert.ToString(value, CultureInfo.InvariantCulture)); return;
        }
        throw new FormatException("Unsupported JSON value " + value.GetType().Name);
    }
    private static void WriteSingle(StringBuilder output, float value)
    {
        if (!Finite(value)) throw new FormatException("Cannot encode a non-finite number.");
        output.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }
    private static void Quote(StringBuilder output, string value)
    {
        output.Append('"');
        foreach (char c in value)
        {
            switch (c)
            {
                case '"': output.Append("\\\""); break;
                case '\\': output.Append("\\\\"); break;
                case '\n': output.Append("\\n"); break;
                case '\r': output.Append("\\r"); break;
                case '\t': output.Append("\\t"); break;
                default:
                    if (c < 32) output.Append("\\u" + ((int)c).ToString("x4"));
                    else output.Append(c);
                    break;
            }
        }
        output.Append('"');
    }
    public static Dictionary<string, object> Decode(string text)
    {
        var parser = new Parser(text); object result = parser.Value(0); parser.Space();
        if (parser.index != text.Length) throw new FormatException("Trailing JSON data.");
        return Object(result);
    }
    private sealed class Parser
    {
        private readonly string text;
        public int index;
        public Parser(string source) { text = source; }
        public void Space() { while (index < text.Length && " \t\r\n".IndexOf(text[index]) >= 0) index++; }
        private char Take() { if (index >= text.Length) throw new FormatException("Incomplete JSON."); return text[index++]; }
        private void Expect(char c) { Space(); if (Take() != c) throw new FormatException("Expected " + c); }
        public object Value(int depth)
        {
            if (depth > 64) throw new FormatException("JSON nesting limit exceeded.");
            Space(); if (index >= text.Length) throw new FormatException("Empty JSON.");
            char c = text[index];
            if (c == '"') return Text();
            if (c == '{')
            {
                index++; var result = new Dictionary<string, object>(); Space();
                if (index < text.Length && text[index] == '}') { index++; return result; }
                while (true)
                {
                    Space(); string key = Text(); Expect(':');
                    if (result.ContainsKey(key)) throw new FormatException("Duplicate JSON key.");
                    result.Add(key, Value(depth + 1)); Space(); c = Take();
                    if (c == '}') return result;
                    if (c != ',') throw new FormatException("Expected comma.");
                }
            }
            if (c == '[')
            {
                index++; var result = new List<object>(); Space();
                if (index < text.Length && text[index] == ']') { index++; return result; }
                while (true)
                {
                    result.Add(Value(depth + 1)); Space(); c = Take();
                    if (c == ']') return result;
                    if (c != ',') throw new FormatException("Expected comma.");
                }
            }
            if (c == 't') { Literal("true"); return true; }
            if (c == 'f') { Literal("false"); return false; }
            if (c == 'n') { Literal("null"); return null; }
            int start = index;
            if (c == '-') index++;
            if (index >= text.Length) throw new FormatException("Invalid number.");
            if (text[index] == '0') index++;
            else { if (text[index] < '1' || text[index] > '9') throw new FormatException("Invalid number."); Digits(); }
            bool real = false;
            if (index < text.Length && text[index] == '.') { real = true; index++; RequiredDigits(); }
            if (index < text.Length && (text[index] == 'e' || text[index] == 'E'))
            { real = true; index++; if (index < text.Length && (text[index] == '+' || text[index] == '-')) index++; RequiredDigits(); }
            string token = text.Substring(start, index - start);
            if (!real && long.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out long integer)) return integer;
            if (!double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out double number) || !Finite(number))
                throw new FormatException("Invalid or non-finite number.");
            return number;
        }
        private void Digits() { while (index < text.Length && text[index] >= '0' && text[index] <= '9') index++; }
        private void RequiredDigits() { int before = index; Digits(); if (before == index) throw new FormatException("Expected digit."); }
        private void Literal(string token) { foreach (char c in token) if (Take() != c) throw new FormatException("Invalid literal."); }
        private string Text()
        {
            if (Take() != '"') throw new FormatException("Expected string.");
            var output = new StringBuilder();
            while (true)
            {
                char c = Take(); if (c == '"') return output.ToString();
                if (c < 32) throw new FormatException("Unescaped control character.");
                if (c != '\\') { output.Append(c); continue; }
                c = Take();
                switch (c)
                {
                    case '"': case '\\': case '/': output.Append(c); break;
                    case 'b': output.Append('\b'); break;
                    case 'f': output.Append('\f'); break;
                    case 'n': output.Append('\n'); break;
                    case 'r': output.Append('\r'); break;
                    case 't': output.Append('\t'); break;
                    case 'u':
                        string hex = ""; for (int i = 0; i < 4; i++) hex += Take();
                        if (!ushort.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ushort code)) throw new FormatException("Invalid Unicode escape.");
                        output.Append((char)code); break;
                    default: throw new FormatException("Invalid escape.");
                }
            }
        }
    }
}
