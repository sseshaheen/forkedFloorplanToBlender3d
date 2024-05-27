import cgi
from cgi import FieldStorage
from threading import Thread
from functools import partial
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer

import json
import logging

from api.post import Post
from api.put import Put
from api.get import Get
from FloorplanToBlenderLib import globalConf

"""
FloorplanToBlender3d
Copyright (C) 2021 Daniel Westberg
"""


class S(BaseHTTPRequestHandler):
    def __init__(self, shared, *args, **kwargs):
        self.shared = shared
        super().__init__(*args, **kwargs)

    def make_client(self):
        client = dict()
        client["address"] = self.address_string()
        client["port"] = str(self.client_address[1])
        return client

    def transform_dict(self, d):
        """Solve issue with all items are lists from query parser!"""
        res = dict()
        for key, item in d.items():
            res[key] = item[0]
        return res

    def query_parser(self, params, rmi):
        """Takes query dict, creates kwargs for methods"""
        function = params["func"]
        out_rmi = rmi(client=self.make_client(), shared_variables=self.shared)
        try:
            argc = getattr(out_rmi, function).__code__.co_argcount
            args = getattr(out_rmi, function).__code__.co_varnames[:argc]
        except Exception:  # Happens if we try to access bad functions!
            return None, None

        # Secure bad requests!
        if "__" in function or (argc == 0 and len(args) == 0):
            return None, None

        # Generate set of correct variables!
        kwargs = dict()

        # If we want to use api_reference variables, these are ignored by swagger
        if "_api_ref" in args:
            kwargs["_api_ref"] = self

        if "_data" in args:
            kwargs["_data"] = params

        if "func" in params:
            kwargs["func"] = params["func"]

        # Add relevant data!
        for parameter in params:
            if parameter in args:
                kwargs[parameter] = params[parameter]
        return out_rmi, kwargs

    def _set_response(self):
        self.send_response(200, "OK")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, PUT, OPTIONS, HEAD, GET")
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def parse_debug_query_params(self):
        parsed_path = urlparse(self.path)
        parsed_data = self.transform_dict(parse_qs(parsed_path.query))
        debug_mode = parsed_data.get('debug', '0') == '1'
        logging_verbose = parsed_data.get('verbose', '0') == '1'
        session_id = parsed_data.get('session_id', None)



        logging.info(f"Debug mode: {debug_mode}, Logging verbose: {logging_verbose}, Session ID: {session_id}")

        globalConf.update_config(debug_mode, logging_verbose, session_id)
        self.configure_logging()

        logging.info(f"Updated globalConf: DEBUG_MODE={globalConf.DEBUG_MODE}, LOGGING_VERBOSE={globalConf.LOGGING_VERBOSE}, DEBUG_SESSION_ID={globalConf.DEBUG_SESSION_ID}")

    def configure_logging(self):
        if globalConf.LOGGING_VERBOSE:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def do_HEAD(self):
        self._set_response()

    def do_OPTIONS(self):
        self._set_response()

    def do_GET(self):
        self.parse_debug_query_params()
        parsed_path = urlparse(self.path)
        kwargs = None
        try:
            parsed_data = self.transform_dict(parse_qs(parsed_path.query))
            rmi, kwargs = self.query_parser(parsed_data, Get)
        except Exception as e:
            message = "RECIEVED GET REQUEST WITH BAD QUERY: " + str(e)
            logging.error(message)
        finally:
            if kwargs is None or rmi is None:
                message = "Function unavailable!"
            else:
                message = getattr(rmi, kwargs["func"])(**kwargs)
        try:
            self._set_response()
            self.wfile.write(bytes(message, encoding="utf-8"))
        except ConnectionAbortedError as e:
            logging.warning("Connection aborted: %s", e)
            return  # This occurs when server is sending file and client isn't waiting for extra message.

    def do_PUT(self):
        self.parse_debug_query_params()
        parsed_path = urlparse(self.path)
        parsed_data = self.transform_dict(parse_qs(parsed_path.query))
        kwargs = None
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])

        if ctype == 'multipart/form-data':
            form = FieldStorage(
                fp=self.rfile, 
                headers=self.headers,
                environ={'REQUEST_METHOD': 'PUT', 'CONTENT_TYPE': self.headers['Content-Type']}
            )
            file_field = form['file']
            if file_field.filename:
                file_content = file_field.file.read()
                try:
                    rmi, kwargs = self.query_parser(parsed_data, Put)
                    if kwargs is None or rmi is None:
                        message = "Function unavailable!"
                    else:
                        kwargs["file"] = file_content
                        (message, _) = getattr(rmi, kwargs["func"])(**kwargs)
                except Exception as e:
                    message = f"Error processing file {file_field.filename}: {str(e)}"
                    logging.error(message)
            else:
                message = "NO FILE PROVIDED!"
                logging.error(message)
        elif ctype in ["html/text", "json/application", "application/json", None]:
            try:
                rmi, kwargs = self.query_parser(parsed_data, Put)
                if kwargs is None or rmi is None:
                    message = "Function unavailable!"
                else:
                    message = getattr(rmi, kwargs["func"])(**kwargs)
            except Exception as e:
                message = f"Error processing PUT request: {str(e)}"
                logging.error(message)
        else:
            message = f"Unsupported content type: {ctype}"
            logging.error(message)

        self._set_response()
        self.wfile.write(bytes(message, encoding='utf-8'))

    def do_POST(self):
        self.parse_debug_query_params()
        if self.headers["Content-Length"]:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            kwargs = None
            try:
                if post_data == bytearray():
                    parsed_data = urlparse(self.path)
                    data = self.transform_dict(parse_qs(parsed_data.query))
                else:
                    data = json.loads(post_data.decode("utf-8"))

                rmi, kwargs = self.query_parser(data, Post)
                if kwargs is None or rmi is None:
                    response = "Function unavailable!"
                else:
                    response = getattr(rmi, kwargs["func"])(**kwargs)
            except ValueError as e:
                response = "RECIEVED POST REQUEST WITH BAD JSON: " + str(e)
                logging.error(response)

        self._set_response()
        self.wfile.write(bytes(response, encoding="utf-8"))


class Server(Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def run(self):
        server_address = (self.shared.restapiHost, int(self.shared.restapiPort))
        httpd = HTTPServer(server_address, partial(S, self.shared))
        try:
            logging.info(
                "REST API SERVER up and serving at %s:%s",
                self.shared.restapiHost,
                self.shared.restapiPort,
            )
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
