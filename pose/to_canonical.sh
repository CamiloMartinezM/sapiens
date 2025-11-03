#!/usr/bin/env bash
# Converts world-space triangulated.ply back to canonical using headpose.txt (canonical->world).
# - Usage: bash to_canonical.sh [PARENT_DIR]
#   If PARENT_DIR is provided (e.g., sub_14/expressions), only its immediate subfolders are processed.
#   If not provided, current directory is used.
# - Uses subfolder's headpose.txt; if missing, falls back to PARENT_DIR/headpose.txt
# - Outputs triangulated_canonical.ply
# Requirements: bash + awk. ASCII PLY only. No exits.

shopt -s nullglob

# Parent directory to scan (default: current dir)
PARENT_DIR="sub_14/expressions"

# Fallback headpose in parent
ROOT_HP="${PARENT_DIR%/}/headpose.txt"

for d in "${PARENT_DIR%/}"/*/ ; do
  [[ -d "$d" ]] || continue
  src="${d%/}/triangulated.ply"
  [[ -f "$src" ]] || { echo "[SKIP] $d : no triangulated.ply"; continue; }

  hp="${d%/}/headpose.txt"
  if [[ ! -f "$hp" ]]; then
    if [[ -f "$ROOT_HP" ]]; then
      hp="$ROOT_HP"
    else
      echo "[SKIP] $d : no headpose.txt (local or parent)"; continue
    fi
  fi

  # Read the 3x4 pose matrix (R|t)
  mapfile -t HP < <(head -n 3 "$hp")
  read r00 r01 r02 tx <<<"${HP[0]}"
  read r10 r11 r12 ty <<<"${HP[1]}"
  read r20 r21 r22 tz <<<"${HP[2]}"

  dst="${d%/}/triangulated_canonical.ply"

  awk -v r00="$r00" -v r01="$r01" -v r02="$r02" \
      -v r10="$r10" -v r11="$r11" -v r12="$r12" \
      -v r20="$r20" -v r21="$r21" -v r22="$r22" \
      -v tx="$tx"  -v ty="$ty"  -v tz="$tz" '
    BEGIN{
      # Invert R (3x3): R^{-1} = adj(R)^T / det(R)
      det = r00*(r11*r22 - r12*r21) - r01*(r10*r22 - r12*r20) + r02*(r10*r21 - r11*r20)
      if (det == 0) { invfail=1 }
      C00 =  (r11*r22 - r12*r21)
      C01 = -(r10*r22 - r12*r20)
      C02 =  (r10*r21 - r11*r20)
      C10 = -(r01*r22 - r02*r21)
      C11 =  (r00*r22 - r02*r20)
      C12 = -(r00*r21 - r01*r20)
      C20 =  (r01*r12 - r02*r11)
      C21 = -(r00*r12 - r02*r10)
      C22 =  (r00*r11 - r01*r10)

      inv00 = C00/det; inv01 = C10/det; inv02 = C20/det;
      inv10 = C01/det; inv11 = C11/det; inv12 = C21/det;
      inv20 = C02/det; inv21 = C12/det; inv22 = C22/det;

      # t_inv = -R^{-1} t
      tinv0 = -(inv00*tx + inv01*ty + inv02*tz)
      tinv1 = -(inv10*tx + inv11*ty + inv12*tz)
      tinv2 = -(inv20*tx + inv21*ty + inv22*tz)

      state="header"; in_vertex=0; vcount=0; row=0;
      xi=yi=zi=-1; prop_idx=0;
    }

    function lower(s){ gsub(/[A-Z]/, "", s); return tolower(s) } # portable-ish
    function isprefix(s,p){ return index(s,p)==1 }

    {
      if (invfail) { print $0; next }

      if (state=="header") {
        line=$0
        low=tolower(line)

        if (isprefix(low, "element vertex")) {
          split(line, a, /[[:space:]]+/); vcount=a[3]+0
          in_vertex=1; prop_idx=0
        } else if (isprefix(low, "element ") && !isprefix(low, "element vertex")) {
          in_vertex=0
        } else if (in_vertex && isprefix(low, "property")) {
          split(line, a, /[[:space:]]+/)
          prop_idx++
          name = a[length(a)]
          if (name=="x") xi=prop_idx
          else if (name=="y") yi=prop_idx
          else if (name=="z") zi=prop_idx
        }

        print line
        if (line=="end_header") {
          state="verts"
          if (xi<0 || yi<0 || zi<0) { state="rest" } # no xyz -> passthrough
        }
        next
      }

      if (state=="verts" && row < vcount) {
        nf = split($0, f, /[[:space:]]+/)

        # Guard: ensure fields exist
        mi = xi; if (yi>mi) mi=yi; if (zi>mi) mi=zi;
        for (i=nf+1; i<=mi; i++) f[i]="0"

        # Read xyz as numbers (supports scientific notation)
        x = f[xi] + 0.0
        y = f[yi] + 0.0
        z = f[zi] + 0.0

        # Xc = R^{-1} Xw + t_inv  ==  R^{-1}(Xw - t)
        xc = inv00*x + inv01*y + inv02*z + tinv0
        yc = inv10*x + inv11*y + inv12*z + tinv1
        zc = inv20*x + inv21*y + inv22*z + tinv2

        f[xi]=sprintf("%.6f", xc)
        f[yi]=sprintf("%.6f", yc)
        f[zi]=sprintf("%.6f", zc)

        out=f[1]
        for (i=2;i<=nf;i++) out = out " " f[i]
        print out

        row++
        if (row==vcount) { state="rest" }
        next
      }

      print $0
    }
  ' "$src" > "$dst" && echo "[DONE] $d -> $(basename "$dst")"
done
