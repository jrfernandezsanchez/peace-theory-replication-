#!/usr/bin/env bash
status=0
for d in */ ; do
  [ -d "$d" ] || continue
  for s in "$d"*.py ; do
    [ -e "$s" ] || continue
    if python3 "$s" > /tmp/out.$$ 2>&1 ; then echo "PASS  $s"
    else echo "FAIL  $s"; cat /tmp/out.$$; status=1; fi
  done
done
rm -f /tmp/out.$$; exit $status
