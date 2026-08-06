FROM node:20-slim

WORKDIR /code

COPY web/package.json web/package-lock.json* ./
RUN npm install

COPY web/ .

EXPOSE 3000

CMD ["npm", "run", "dev"]
