test_mask_1 <- array(c(255, 0, 0, 0, 111, 0, 111, 0, 222,
                       255, 0, 0, 0, 111, 0, 111, 0, 222,
                       255, 0, 0, 0, 111, 0, 111, 0, 222), dim = c(3, 3, 3))
test_mask_2 <- array(c(255, 0, 0, 0, 111, 111, 0, 0, 222,
                       255, 0, 0, 0, 111, 111, 0, 0, 222,
                       255, 0, 0, 0, 111, 111, 0, 0, 222), dim = c(3, 3, 3))
test_mask_3 <- array(c(222, 0, 0, 0, 111, 0, 111, 0, 255,
                       222, 0, 0, 0, 111, 0, 111, 0, 255,
                       222, 0, 0, 0, 111, 0, 111, 0, 255), dim = c(3, 3, 3))
for (i in 1:3) {
  filename_img = paste0("inst/testdata/dir/images/test_image_", i, ".png")
  filename_mask = paste0("inst/testdata/dir/masks/test_mask_", i, ".png")
  png(filename = filename_img)
  grid.raster(get(paste0("test_mask_", i)) / 255)
  dev.off()
  png(filename = filename_mask)
  grid.raster(get(paste0("test_mask_", i)) / 255)
  dev.off()
}

test_mask_1_1 <- array(c(0, 0, 0, 0, 111, 0, 111, 0, 0,
                             0, 0, 0, 0, 111, 0, 111, 0, 0,
                             0, 0, 0, 0, 111, 0, 111, 0, 0), dim = c(3, 3, 3))
test_mask_1_2 <- array(c(0, 0, 0, 0, 0, 0, 0, 0, 222,
                             0, 0, 0, 0, 0, 0, 0, 0, 222,
                             0, 0, 0, 0, 0, 0, 0, 0, 222), dim = c(3, 3, 3))
test_mask_1_3 <- array(c(255, 0, 0, 0, 0, 0, 0, 0, 0,
                             255, 0, 0, 0, 0, 0, 0, 0, 0,
                             255, 0, 0, 0, 0, 0, 0, 0, 0), dim = c(3, 3, 3))

test_mask_2_1 <- array(c(0, 0, 0, 0, 111, 111, 0, 0, 0,
                         0, 0, 0, 0, 111, 111, 0, 0, 0,
                         0, 0, 0, 0, 111, 111, 0, 0, 0), dim = c(3, 3, 3))
test_mask_2_2 <- array(c(0, 0, 0, 0, 0, 0, 0, 0, 222,
                         0, 0, 0, 0, 0, 0, 0, 0, 222,
                         0, 0, 0, 0, 0, 0, 0, 0, 222), dim = c(3, 3, 3))
test_mask_2_3 <- array(c(255, 0, 0, 0, 0, 0, 0, 0, 0,
                         255, 0, 0, 0, 0, 0, 0, 0, 0,
                         255, 0, 0, 0, 0, 0, 0, 0, 0), dim = c(3, 3, 3))

test_mask_3_1 <- array(c(0, 0, 0, 0, 111, 0, 111, 0, 0,
                         0, 0, 0, 0, 111, 0, 111, 0, 0,
                         0, 0, 0, 0, 111, 0, 111, 0, 0), dim = c(3, 3, 3))
test_mask_3_2 <- array(c(0, 0, 0, 0, 0, 0, 0, 0, 255,
                         0, 0, 0, 0, 0, 0, 0, 0, 255,
                         0, 0, 0, 0, 0, 0, 0, 0, 255), dim = c(3, 3, 3))
test_mask_3_3 <- array(c(222, 0, 0, 0, 0, 0, 0, 0, 0,
                         222, 0, 0, 0, 0, 0, 0, 0, 0,
                         222, 0, 0, 0, 0, 0, 0, 0, 0), dim = c(3, 3, 3))

for (i in 1:3) {
  filename_img = paste0("inst/testdata/nested_dirs/image_", i, "/images/test_image_", i, ".png")
  png(filename = filename_img)
  grid.raster(get(paste0("test_mask_", i)) / 255)
  dev.off()
  for (j in 1:3) {
    filename_mask = paste0("inst/testdata/nested_dirs/image_", i, "/masks/test_mask_", i, "_", j, ".png")
    png(filename = filename_mask)
    grid.raster(get(paste0("test_mask_", i, "_", j)) / 255)
    dev.off()
  }
}
