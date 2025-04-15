# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Hitalo M. <https://github.com/HitaloM>

import logging
import time

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.policies import RetryPolicy

logger = logging.getLogger(__name__)


class CustomRetryPolicy(RetryPolicy):
    """
    Custom retry policy for handling rate limits and transient errors.

    Extends the RetryPolicy to provide specialized handling for HTTP 429 errors
    (Too Many Requests) and other transient Azure service errors.
    """

    async def send(self, request: PipelineRequest) -> PipelineResponse:
        """
        Send request with retry handling for API rate limits and failures.

        Args:
            request: Pipeline request to send

        Returns:
            Response from the pipeline

        Raises:
            ClientAuthenticationError: When authentication fails
            AzureError: For other errors after exceeding retries
        """
        retry_settings = self.configure_retries(request.context.options)
        self._configure_positions(request, retry_settings)
        absolute_timeout = retry_settings["timeout"]
        is_response_error = True
        response = None

        while True:
            start_time = time.time()
            transport = request.context.transport

            try:
                self._configure_timeout(request, absolute_timeout, is_response_error)
                request.context["retry_count"] = len(retry_settings["history"])
                response = await self.next.send(request)  # type: ignore

                if response.http_response.status_code == 429:
                    logger.warning("HTTP 429 Too Many Requests encountered.")
                    raise HttpResponseError(
                        message="Too many requests", response=response.http_response
                    )

                if self.is_retry(retry_settings, response) and self.increment(
                    retry_settings, response=response
                ):
                    await self.sleep(retry_settings, transport, response=response)  # type: ignore
                    is_response_error = True
                    continue

                break

            except ClientAuthenticationError as e:
                logger.error("Client authentication failed: %s", e)
                raise

            except AzureError as err:
                if (
                    absolute_timeout > 0
                    and self._is_method_retryable(retry_settings, request.http_request)
                    and self.increment(retry_settings, response=request, error=err)
                ):
                    await self.sleep(retry_settings, transport)  # type: ignore
                    is_response_error = not isinstance(err, ServiceRequestError)
                    continue
                logger.error("Azure error encountered: %s", err)
                raise

            finally:
                elapsed = time.time() - start_time
                if absolute_timeout:
                    absolute_timeout -= elapsed

        if response is None:
            msg = "Maximum retries exceeded."
            logger.error(msg)
            raise AzureError(msg)

        self.update_context(response.context, retry_settings)
        return response
