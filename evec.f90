	program main
	integer i,j,k,l,f
	real a,b,c,d,e
	real sca
	integer nca,id
	character pl*80
	real,allocatable:: cod(:,:), vc(:,:)
	nca=291
	nc=1
	sca=70.
	allocate(vc(nca,3),cod(nca,3))


	open(1,file='ca')
	do i=1,nca
	read(1,'(a80)') pl
	read(pl(30:54),*) (cod(i,k),k=1,3)
	enddo
	close(1)

	do f=1,10

	nc=f
	open(10,file='evecs3.dat')
	do i=1,9999999
	read(10,'(a80)') pl
	if(pl(2:5).eq.'****') then
	read(10,*) id
	if(id.eq.nc) then
	read(10,'(7f11.5)') ((vc(j,k),k=1,3),j=1,nca)
	goto 100
	endif
	endif
	enddo
100	continue
	close(10)


	do i=1,nca
	write(70+f,'(a13,3f10.5,a5,3f10.5,a3)') 'draw arrow { ',(cod(i,k),k=1,3),' } { ', (cod(i,k)+vc(i,k)*sca,k=1,3) , ' }'
	enddo

	enddo
	end
