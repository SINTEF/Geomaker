PYUIC ?= pyuic5
PYRCC ?= pyrcc5
PYUIC_FLAGS ?= --from-imports
PYRCC_FLAGS ?=

INDIR = gui
OUTDIR = geomaker/ui

GUI_FILES = interface.ui thumbnail.ui jobdialog.ui
RESOURCE_FILE = ${INDIR}/resources.qrc
RESOURCE_OUT = ${OUTDIR}/resources_rc.py

.PHONY: all clean

all: $(GUI_FILES:%.ui=${OUTDIR}/%.py) ${RESOURCE_OUT}

${RESOURCE_OUT}: ${RESOURCE_FILE}
	${PYRCC} ${PYRCC_FLAGS} $< -o $@

${OUTDIR}/%.py: ${INDIR}/%.ui
	${PYUIC} ${PYUIC_FLAGS} $< -o $@
