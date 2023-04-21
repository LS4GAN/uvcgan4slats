#!/usr/bin/env bash

DATAROOT="${UVCGAN_DATA:-data}"
DATADIR="${DATAROOT}/slats"


declare -A URL_LIST=(
    [tiles]="https://zenodo.org/record/7809108/files/slats_tiles.zip"
    [center_crops]="https://zenodo.org/record/7809108/files/slats_center_crops.zip"
)

declare -A CHECKSUMS=(
    [tiles]="fcab8ff8525bcb00a9aa33b604fb11ff"
    [center_crops]="2a064cd4fd0b7a1bb8142c1846d03e9e"
)

die ()
{
    echo "${*}"
    exit 1
}

usage ()
{
    cat <<EOF
USAGE: download_slats_datasets.sh DATASET
where DATASET is one of tiles and center_crops.
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
    local url="${1}"
    local fname="${2}"
    local checksum="${3}"

    exec_or_die mkdir -p "${DATADIR}"

    local path="${DATADIR}/${fname}"

    if [[ ! -e "${path}" ]]
    then
        exec_or_die wget "${url}" --output-document "${path}"
    fi

    if [[ -n "${checksum}" ]]
    then
        # shellcheck disable=SC2155
        local test_csum="$(calc_md5_hash "${path}")"

        if [[ "${test_csum}" == "${checksum}" ]]
        then
            echo " - Checksum valid"
        else
            die "Checksum mismatch for '${path}' ${test_csum} vs ${checksum}"
        fi
    fi
}

download_and_extract_zip ()
{
    local url="${1}"
    local zip="${2}"
    local checksum="${3}"

    download_and_check  "${url}" "${zip}" "${checksum}"
    exec_or_die unzip "${DATADIR}/${zip}" -d "${DATADIR}"

    exec_or_die rm -f "${DATADIR}/${zip}"

    echo " - Dataset is unpacked to '${path}'"
}

check_dset_exists ()
{
    local path="${1}"

    if [[ -e "${path}" ]]
    then

        read -r -p "Dataset '${path}' exists. Overwrite? [yN]: " ret
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

download_tiles ()
{
    local url="${URL_LIST["tiles"]}"
    local zip="slats_tiles.zip"
    local path="${DATADIR}/slats_tiles"

    check_dset_exists "${path}"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[tiles]}"
}

download_center_crops ()
{
    local url="${URL_LIST["center_crops"]}"
    local zip="slats_center_crops.zip"
    local path="${DATADIR}/slats_center_crops"

    check_dset_exists "${path}"

    download_and_extract_zip "${url}" "${zip}" "${CHECKSUMS[center_crops]}"
}


dataset="${1}"

case "${dataset}" in
    tiles)
        download_tiles
        ;;
    center_crops)
        download_center_crops
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage "Unknown dataset '${dataset}'"
esac
