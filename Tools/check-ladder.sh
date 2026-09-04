#!/bin/zsh
#
# Checks that the model ladder is the same in every place that names it.
#
# There are four such places, and they have drifted before with no symptom that
# any test or build could catch:
#
#   Lucid/Metal/LearnedUpscaler.swift   variants[] — what the app will load
#   Tools/release.sh                    which packages go into the .app
#   Tools/run-poc.sh                    which packages go into a dev build
#   .gitignore                          which packages are tracked at all
#
# When run-poc.sh kept compiling the ch28 ladder after the app moved to ch32u,
# every build succeeded, the app launched, the bridge connected, the page sent
# frames, and nothing was ever enhanced: LearnedUpscaler.model() looked for a
# resource that was not installed and threw, and there is no fallback upscaler
# by design. Nothing failed loudly. This script is the thing that fails loudly.
#
#   Tools/check-ladder.sh
#
set -euo pipefail

repo="${0:A:h:h}"
cd "$repo"
failures=0

fail() { print -u2 "  ✗ $1"; failures=1 }
pass() { print "  ✓ $1" }

# The stem the app actually loads, taken from the source rather than assumed.
stem="$(grep -oE 'SPAN_x4_[A-Za-z0-9]+_' Lucid/Metal/LearnedUpscaler.swift | head -1 | sed 's/_$//')"
[[ -n "$stem" ]] || { print -u2 "cannot find the model name stem in LearnedUpscaler.swift"; exit 1 }
print "model stem: $stem"

# The ladder, from LearnedUpscaler.variants.
typeset -a want
want=(${(f)"$(grep -oE 'Variant\(width: *[0-9]+, *height: *[0-9]+' Lucid/Metal/LearnedUpscaler.swift \
  | sed -E 's/.*width: *([0-9]+).*height: *([0-9]+)/\1x\2/')"})
(( ${#want} )) || { print -u2 "cannot parse LearnedUpscaler.variants"; exit 1 }
print "ladder:     ${want[*]}  (${#want} sizes)"
print

# Each of the other three places must name exactly the same set.
check_list() {
  local label="$1" file="$2"
  shift 2
  typeset -a got
  got=("$@")
  if [[ "${(j: :)${(o)want}}" == "${(j: :)${(o)got}}" ]]; then
    pass "$label agrees ($file)"
  else
    fail "$label disagrees ($file)"
    print -u2 "      want: ${(j:, :)${(o)want}}"
    print -u2 "      got:  ${(j:, :)${(o)got}}"
  fi
}

check_list "release.sh ladder" Tools/release.sh \
  ${=$(grep -oE 'for size in [0-9x ]+' Tools/release.sh | head -1 | sed 's/for size in //')}

check_list "run-poc.sh ladder" Tools/run-poc.sh \
  ${=$(grep -oE 'for size in [0-9x ]+' Tools/run-poc.sh | head -1 | sed 's/for size in //')}

check_list ".gitignore tracked set" .gitignore \
  ${(f)"$(grep -oE "^!Model/${stem}_[0-9]+x[0-9]+\.mlpackage" .gitignore \
    | sed -E "s|^!Model/${stem}_||; s|\.mlpackage$||")"}

# The stem must match everywhere too, not just the sizes.
for file in Tools/release.sh Tools/run-poc.sh .gitignore; do
  if grep -qE "SPAN_x4_[A-Za-z0-9]+_" "$file"; then
    other="$(grep -oE 'SPAN_x4_[A-Za-z0-9]+_' "$file" | sort -u | sed 's/_$//' | tr '\n' ' ')"
    if [[ "${other% }" == "$stem" ]]; then
      pass "$file uses $stem"
    else
      fail "$file names ${other% } but the app loads $stem"
    fi
  fi
done

# And the packages the app loads have to actually be on disk and tracked, or a
# fresh clone builds an app that silently enhances nothing.
print
for size in $want; do
  package="Model/${stem}_${size}.mlpackage"
  if [[ ! -d "$package" ]]; then
    fail "$package is missing from the working tree"
  elif ! git ls-files --error-unmatch "$package" >/dev/null 2>&1; then
    fail "$package exists but is not tracked by git - a clone will not have it"
  else
    pass "$package tracked"
  fi
done

print
if (( failures )); then
  print -u2 "ladder check FAILED"
else
  print "ladder check passed"
fi
exit $failures
