from tiling import reconstruir_geotiff
from tiling import crop_geotiff
from tiling import dividir_treino_teste_geotiff
from tiling import sobrepor_mascara_verde


# reconstruir_geotiff("output_masks", "reconstrucao_masks_teste.tif", tile_size=256, overlap=0.0)
# sobrepor_mascara_verde("plantacao_teste.tif", "reconstrucao_masks_teste.tif", "sobreposicao.tif", alpha=0.4)


reconstruir_geotiff("output_masks_plantacao2", "reconstrucao_masks_plantacao2.tif", tile_size=256, overlap=0.0)
sobrepor_mascara_verde("plantacao2.png", "reconstrucao_masks_plantacao2.tif", "sobreposicao.tif", alpha=0.4)