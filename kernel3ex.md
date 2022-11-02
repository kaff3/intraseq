## Kernels 3 + 4 example
vi har to sorterede blocks med værdier fra 0-3:
	b1: 	0 0 0 1 2 3
	b2: 	1 2 2 2 3 3

histogrammerne er derfor: 
	b1 hist: 3 1 1 1 
	b2 hist: 0 1 3 2
	  
I hukommelsen ligger de transponerede, så efter digit:
	3 0 1 1 1 3 1 2

vi laver så et globalt excl scan:
	0 3 3 4 5 6 9 10

dette svarer derfor til følgende globale positioner:
	b1 global: 0 3 5 9
	b2 global: 3 4 6 10

Vi skal så finde de lokale positioner. Vi har de scannede lokale histogrammer fra kernel 1: 
	b1 scan: 0 3 4 5 (b1 hist scanned)
	b2 scan: 0 0 1 4 (b2 hist scanned)

vi finder så de lokale positioner, hvis count != 1 (kan enten læses fra b_ hist eller ved at trække næste element i det scannede fra), hvis count == 1, så er det bare plads 0:
	b1: 		0 0 0 1 2 3
	count:		3 3 3 1 1 1
	 
	index:		0 1 2   
			  - 
	scan val	0 0 0 
			  =
	b1 local:   0 1 2 0 0 0

	
	b2: 		1 2 2 2 3 3
	count:		1 3 3 3 2 2

	index:		  1 2 3 4 5	
			  - 
	scan val	  1 1 1 4 4
		      =
	b2 local:   0 0 1 2 0 1

nu kan vi sortere ved scatter:

	b1: 		0 0 0 1 2 3
	
	b1 local:   0 1 2 0 0 0
			  +
	b1 global:  0 0 0 3 5 9
			  =
	pos: 		0 1 2 3 5 9 	

	index:			 0, 1, 2, 3, 4, 5, 6, 7
	sorteret array: [0, 0, 0, 1, _, 2, _, _, _, 3, _, _]

	b2: 		1 2 2 2 3  3
	
	b2 local:   0 0 1 2 0  1
			  +
	b2 global:  4 6 6 6 10 10
			  =
	pos: 		4 6 7 8 10 11 	
	
	index:			 0, 1, 2, 3, 4, 5, 6, 7
	sorteret array: [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3]
