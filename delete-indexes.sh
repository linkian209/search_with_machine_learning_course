# WARNING: this will silently delete both of your indexes

curl -k -X DELETE -u admin  "https://localhost:9200/bbuy_products"
<<<<<<< HEAD
echo "\n"
curl -k -X DELETE -u admin  "https://localhost:9200/bbuy_queries"
echo "\n"
=======
curl -k -X DELETE -u admin  "https://localhost:9200/bbuy_queries"
curl -k -X DELETE -u admin  "https://localhost:9200/bbuy_annotations"
>>>>>>> 848fab8940abc2d7a8c2afaad25538df31461976
