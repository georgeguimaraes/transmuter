defmodule Transmuter.HubError do
  @moduledoc """
  Raised when the Hugging Face Hub gives an error status for a repository or one of its files, such as 401 for a gated repository without a token, or 404 for a repository that does not exist.

  ## Fields

    * `:repo` - the repository
    * `:status` - the HTTP status
    * `:details` - the file, or the body of the response of the Hub API
  """

  defexception [:repo, :status, :details]

  @impl true
  def message(error) do
    "the Hugging Face Hub gave status #{error.status} for #{error.repo}: #{error.details}. " <>
      "Set HF_TOKEN or run hf auth login for a private or gated repository"
  end
end
