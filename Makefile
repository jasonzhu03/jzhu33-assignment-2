# Install necessary dependencies using pip and create a virtual environment
install:
	@echo "Creating a virtual environment and installing dependencies..."
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Dependencies installed."

# Run the web application on localhost:3000
run:
	@echo "Activating virtual environment and starting the web server..."
	. venv/bin/activate && FLASK_APP=app.py flask run --host=0.0.0.0 --port=3000