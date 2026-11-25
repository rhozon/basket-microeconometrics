# Download The Complete Journey (84.51 degrees) through the completejourney package (CC0)
# and write plain CSV files. Promotions are aggregated to category, store and week here,
# because the full table has tens of millions of rows.
out <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out)) out <- "data_raw"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
lib <- Sys.getenv("R_LIBS_USER")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
for (p in c("completejourney", "dplyr")) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({ library(completejourney); library(dplyr) })
tx <- get_transactions(verbose = FALSE)
pr <- get_promotions(verbose = FALSE)
write.csv(tx, gzfile(file.path(out, "transactions.csv.gz")), row.names = FALSE)
for (nm in c("products", "demographics", "campaigns", "campaign_descriptions", "coupons", "coupon_redemptions")) {
  write.csv(get(nm), gzfile(file.path(out, paste0(nm, ".csv.gz"))), row.names = FALSE)
}
promo <- pr %>%
  inner_join(products %>% select(product_id, product_category), by = "product_id") %>%
  group_by(store_id, week, product_category) %>%
  summarise(n_promo_items = n(),
            n_display = sum(display_location != "0"),
            n_mailer = sum(mailer_location != "0"), .groups = "drop")
write.csv(promo, gzfile(file.path(out, "promotions_by_category.csv.gz")), row.names = FALSE)
cat("transactions", nrow(tx), "promotions", nrow(pr), "promo cells", nrow(promo), "\n")
