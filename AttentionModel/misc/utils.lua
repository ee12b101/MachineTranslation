local utils = {}

function utils.getopt(opt, key, defaultValue)
	if defaultValue == nil and (opt == nil or opt[key] == nil) then
		print('key: ' .. key .. ' not found in opt.')
	end
	if opt == nil or opt[key] == nil then return defaultValue end
	return opt[key]
end

return utils
