import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filename='/home/hanyuji/projects/PJ4/demo.log',
)

for i in range(500):
    if i % 20 == 1:
        logging.info(f'------{i}')
