library(bio3d)
traj <- read.ncdf("nowatr.nc")
pdb <- read.pdb("../comp.pdb")
ca.inds <- atom.select(pdb, elety="CA")
xyz <- fit.xyz(fixed=pdb$xyz, mobile=traj, fixed.inds=ca.inds$xyz, mobile.inds=ca.inds$xyz)
dim(xyz) == dim(traj)
pc <- pca.xyz(xyz[,ca.inds$xyz])
png("pca.png")
par(
    cex.axis=1.1,   
    font.axis=2,    # 轴刻度的字体加粗
    font.lab=2      # 轴标签的字体加粗
)
plot(pc,col=bwr.colors(nrow(xyz)))
dev.off()
hc <- hclust(dist(pc$z[,1:2]))
grps1 <- cutree(hc, k=4)
png("clust4.png")
par(
    cex.axis=1.1,   
    font.axis=2,    # 轴刻度的字体加粗
    font.lab=2      # 轴标签的字体加粗
)
plot(pc, col=grps1)
dev.off()
grps2 <- cutree(hc, k=2)
png("clust2.png")
par(
    cex.axis=1.1,   
    font.axis=2,    # 轴刻度的字体加粗
    font.lab=2      # 轴标签的字体加粗
)
plot(pc, col=grps2)
dev.off()
pc1 <- mktrj.pca(pc, pc=1, b=pc$au[,1], file="pc1.pdb")
pc2 <- mktrj.pca(pc, pc=2, b=pc$au[,2], file="pc2.pdb")
write.ncdf(pc1, "trj_1.nc")
write.ncdf(pc2, "trj_2.nc")
cij <- dccm(xyz[,ca.inds$xyz])
print(cij)
png("corr.png")
plot(cij)
write.csv(cij, file = "cij_matrix.csv", row.names = FALSE)

# 获取当前x轴和y轴的刻度值
#x_ticks <- axTicks(1)  # x轴刻度
#y_ticks <- axTicks(2)  # y轴刻度

# 在现有刻度值的基础上加10并重新绘制轴刻度
#axis(1, at = x_ticks, labels = x_ticks + offset)  # x轴刻度值加10
#axis(2, at = y_ticks, labels = y_ticks + offset)  # y轴刻度值加10
dev.off()
#pymol.dccm(cij,pdb,type="launch")

