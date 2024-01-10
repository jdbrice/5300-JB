
DATE=$(shell date '+%Y-%m-%d')

build: 
	jupyter-book build .

deploy: build
	tar -czvf ${DATE}.tar.gz _build
	cp -r _build/html/ ${HOME}/Development/jdbburg/static/classes/Sp24Phy5300-JB/