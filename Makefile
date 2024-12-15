################################################################################
#
# Makefile for each test program
#
################################################################################

PWD         = $(shell pwd)
TOPDIR      = $(abspath ../../../../../..)
TESTDIR     = ../../../..


# user-defined source files
CODE_DIR    = .
ORG_SRCS    =

# parameters for run
SNOCONF     = sno.vgridope.d01.conf,sno.hgridope.d01.conf

TPROC       = 4,1

# required data (parameters,distributed files)
DATDIR      =
DATPARAM    =
DATDISTS    =

# build, makedir, run, jobshell, allclean, clean is inside of common Makefile
include $(TESTDIR)/Makefile.common


all: run

