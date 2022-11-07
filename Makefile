
all: radix radix-validate

dirs:
	mkdir -p bin
	kmdir -p data

radix: dirs main.cu
	nvcc main.cu -o bin/radix

radix-validate: dirs main.cu
	nvcc main.cu -DRADIX_VALIDATE -o bin/radix-validate

fut-bench: radix-fut.fut
	futhark bench --backend=cuda radix-fut.fut

clean:
	rm -f bin/radix
	rm -f bin/radix-validate

