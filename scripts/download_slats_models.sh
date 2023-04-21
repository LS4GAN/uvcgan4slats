#!/usr/bin/env bash

OUTROOT="${UVCGAN_OUTDIR:-outdir}"
OUTDIR="${OUTROOT}/slats/pretrained"
ZENODO_BASE="https://zenodo.org/record/7853835/"

declare -A CHECKSUMS=(
    [config.json]=4cb0a39f87c36af28766a63188f55591
    [net_gen_ab.pth]=416abd1c9e113bca24509484ddc3f466
    [net_gen_ba.pth]=b57580f84577bf4ff3a271497b373cf3
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_slats_models.sh [-h|--help].

Download UVCGAN generator (or translator) models pre-trained on the slats tiles.

EOF

    if [[ $# -gt 0 ]]
    then
        die "${*}"
    else
        exit 0
    fi
}

exec_or_die ()
{
    "${@}" || die "Failed to execute: '${*}'"
}

calc_md5_hash ()
{
    local path="${1}"
    md5sum "${path}" | cut -d ' ' -f 1 | tr -d '\n'
}

download_and_check ()
{
    local filename="${1}"
    local dest="${OUTDIR}/${filename}"
    local url="${ZENODO_BASE}/files/${filename}"
    exec_or_die wget "${url}" --output-document "${dest}"

    local null_csum="${CHECKSUMS[${filename}]}"

    if [[ -n "${null_csum}" ]]
    then
        local test_csum="$(calc_md5_hash "${dest}")"

        if [[ "${test_csum}" == "${null_csum}" ]]
        then
            echo " - Checksum valid"
        else
            die "Checksum mismatch for '${dest}' "\
                "${test_csum} vs ${null_csum}"
        fi
    fi
}

check_model_exists ()
{
    local path="${1}"

    if [[ -e "${path}" ]]
    then
        read -r -p "Model '${path}' exists. Overwrite? [yN]: " ret
        case "${ret}" in
            [Yy])
                exec_or_die rm -rf "${path}"
                ;;
            *)
                exit 0
                ;;
        esac
    fi
}

download_slats_models ()
{
    local folder="${OUTDIR}"

    check_model_exists "${folder}"

    if [[ ! -e "${folder}" ]]
    then
        exec_or_die mkdir -p "${folder}"
    fi

    download_and_check "config.json"
    download_and_check "net_gen_ab.pth"
    download_and_check "net_gen_ba.pth"
}

while [ $# -gt 0 ]
do
    case "$1" in
        -h|--help|help)
            usage
            ;;
        *)
            usage "Unknown model '$1'"
    esac
done

download_slats_models
