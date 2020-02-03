PYUIC ?= pyuic5
PYRCC ?= pyrcc5
PYUIC_FLAGS ?= --from-imports
PYRCC_FLAGS ?=

OUTDIR = geomaker/ui

GUI_FILES = interface.ui thumbnail.ui jobdialog.ui
RESOURCE_FILE = resources.qrc
RESOURCE_OUT = ${OUTDIR}/resources_rc.py

.PHONY: all

all: $(GUI_FILES:%.ui=${OUTDIR}/%.py) ${RESOURCE_OUT}

${RESOURCE_OUT}: ${RESOURCE_FILE}
	${PYRCC} ${PYRCC_FLAGS} $< -o $@

${OUTDIR}/%.py: %.ui
	${PYUIC} ${PYUIC_FLAGS} $< -o $@
