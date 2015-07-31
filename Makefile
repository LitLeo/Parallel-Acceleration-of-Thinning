ALGNAMES := Image Thinning_gpu Thinning_gpu_pt Thinning_gpu_pt_con Thinning_gpu_four \
		    Thinning_gpu_pt_con_four
			
INCFILES := ErrorCode.h $(addsuffix .h, $(ALGNAMES))
OBJFILES := main.o $(addsuffix .o, $(ALGNAMES))

EXEFILE  := okanoexec

NVCCCMD  := nvcc
NVCCFLAG := -arch=sm_20
NVLDFLAG := -lnppi

world: $(EXEFILE)

$(EXEFILE): $(OBJFILES)
	$(NVCCCMD) $(OBJFILES) -o $(EXEFILE) $(NVLDFLAG)

$(OBJFILES): %.o:%.cu $(INCFILES)
	$(NVCCCMD) -c $(filter %.cu, $<) -o $@ $(NVCCFLAG)

clean:
	rm -rf $(OBJFILES) $(EXEFILE)
