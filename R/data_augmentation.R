# flip_image <- function(image) {
#   tf$image$flip_left_right(image)$numpy()
# }
#
# flip_boxes <- function(boxes) {
#   boxes %>%
#     mutate_at(vars("xmin", "xmax"), ~ 1 - .)
# }
#
# apply_affine_transform_to_image <- function(image, theta = 0, tx = 0, ty = 0, shear = 0, zx = 1, zy = 1) {
#   tf$keras$preprocessing$image$apply_affine_transform(image, theta, ty, tx, shear, zx, zy)
# }
#
# move_boxes <- function(boxes, height, width, tx = 0, ty = 0) {
#   boxes %>%
#     mutate_at(vars("xmin", "xmax"), ~ . - tx / width) %>%
#     mutate_at(vars("ymin", "ymax"), ~ . - ty / height) %>%
#     rowwise() %>%
#     mutate_at(vars("xmin", "ymin", "xmax", "ymax"), ~ max(., 0)) %>%
#     mutate_at(vars("xmin", "ymin", "xmax", "ymax"), ~ min(., 1)) %>%
#     filter(xmax > xmin, ymax > ymin)
# }
#
# rotate_boxes <- function(boxes, height, width, theta = 0) {
#   boxes %>%
#     mutate_at(vars("xmin", "xmax"), ~ . * width) %>%
#     mutate_at(vars("ymin", "ymax"), ~ . * height) %>%
#     pmap_df(function(label, xmin, ymin, xmax, ymax) {
#       x1 <- cos(-theta) * (xmin - width / 2) - sin(-theta) * (ymin - height / 2) + width / 2
#       x2 <- cos(-theta) * (xmin - width / 2) - sin(-theta) * (ymax - height / 2) + width / 2
#       x3 <- cos(-theta) * (xmax - width / 2) - sin(-theta) * (ymin - height / 2) + width / 2
#       x4 <- cos(-theta) * (xmax - width / 2) - sin(-theta) * (ymax - height / 2) + width / 2
#       y1 <- sin(-theta) * (xmin - width / 2) + cos(-theta) * (ymin - height / 2) + height / 2
#       y2 <- sin(-theta) * (xmin - width / 2) + cos(-theta) * (ymax - height / 2) + height / 2
#       y3 <- sin(-theta) * (xmax - width / 2) + cos(-theta) * (ymin - height / 2) + height / 2
#       y4 <- sin(-theta) * (xmax - width / 2) + cos(-theta) * (ymax - height / 2) + height / 2
#       tibble(label = label,
#              xmin = min(c(x1, x2, x3, x4)) / width,
#              ymin = min(c(y1, y2, y3, y4)) / height,
#              xmax = max(c(x1, x2, x3, x4)) / width,
#              ymax = max(c(y1, y2, y3, y4)) / height)
#     })
# }
#
# shear_boxes <- function(boxes, height, width, shear = 0) {
#   boxes %>%
#     mutate(ymin = ymin - tan(shear) * xmin,
#            ymax = ymax + tan(shear) * xmax)
# }
#
# apply_affine_transform_to_boxes <- function(boxes, height, width, theta = 0, tx = 0, ty = 0, shear = 0, zx = 1, zy = 1) {
#   boxes %>%
#     move_boxes(height, width, tx, ty) %>%
#     rotate_boxes(height, width, theta) %>%
#     shear_boxes(height, width, shear)
# }
