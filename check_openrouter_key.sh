#!/usr/bin/env bash
# Verifies an OpenRouter key works, without ever printing it.
KEY="$(cat ~/.openrouter_key 2>/dev/null || echo "$OPENROUTER_API_KEY")"
[ -z "$KEY" ] && { echo "no key found in ~/.openrouter_key or \$OPENROUTER_API_KEY"; exit 1; }
echo "key length: ${#KEY} chars, starts ${KEY:0:8}..."
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $KEY" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d: print('INVALID:', d['error'].get('message')); raise SystemExit(1)
k=d.get('data',{})
lim=k.get('limit'); used=k.get('usage')
print(f\"VALID. usage so far \${used}\" + (f\", limit \${lim}\" if lim else ', no limit'))
"
