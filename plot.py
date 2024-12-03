import matplotlib.pyplot as plt
plt.imshow(saliency_map, vmin=0.15, vmax=0.45, cmap='jet')
cbar = plt.colorbar()
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_weight('bold')
    l.set_fontsize(24)

plt.xlabel('Residue', fontsize=24, rotation=0, fontweight='bold')
plt.ylabel('Residue', fontsize=24, rotation=90, fontweight='bold')
plt.xticks(fontsize=24, fontweight='bold')
plt.yticks(fontsize=24, fontweight='bold')
plt.savefig('/import-contact/contact.jpg')