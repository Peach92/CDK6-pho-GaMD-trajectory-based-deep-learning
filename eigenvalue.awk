	BEGIN {
	  NF=" "
	}
	NF <= 2 && $1 !~/\*/ {
	  print  $0
	}
